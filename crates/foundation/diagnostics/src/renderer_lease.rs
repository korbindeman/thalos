//! Machine-wide ownership of the workstation GPU renderer.
//!
//! A tile-only budget cannot make two full renderers safe: each process also
//! owns textures, shadows, clouds, pipelines, mesh slabs, and transient boot
//! allocations. Thalos therefore permits exactly one canonical game-shaped
//! renderer per machine. The operating-system primitive is the authority; the
//! JSON owner record exists only to make a refusal actionable.

use std::{
    fmt, fs,
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

const OWNER_DIR: &str = "thalos-machine-renderer";
const OWNER_FILE: &str = "owner.json";
#[cfg(windows)]
const RENDERER_MUTEX_NAME: &str = "Global\\ThalosMachineRendererV1";
/// Distinct from capture-invalid (3): no renderer was initialized.
pub const EXIT_RENDERER_BUSY: i32 = 4;

/// The two canonical processes capable of owning the workstation renderer.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RendererRole {
    InteractiveGame,
    CaptureHost,
}

impl RendererRole {
    pub const fn label(self) -> &'static str {
        match self {
            Self::InteractiveGame => "interactive game",
            Self::CaptureHost => "capture host",
        }
    }
}

impl fmt::Display for RendererRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// Human-readable metadata published while the OS lease is held.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RendererOwner {
    pub pid: u32,
    pub role: RendererRole,
    pub started_unix_ms: u128,
    pub command: String,
    pub workspace: String,
}

impl RendererOwner {
    fn current(role: RendererRole) -> Self {
        Self {
            pid: std::process::id(),
            role,
            started_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            command: std::env::args().collect::<Vec<_>>().join(" "),
            workspace: std::env::current_dir()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| "<unknown>".into()),
        }
    }
}

/// A renderer launch refused before Bevy/wgpu is initialized.
#[derive(Debug)]
pub enum RendererLeaseError {
    Busy {
        requested: RendererRole,
        owner: Option<RendererOwner>,
    },
    Io(io::Error),
}

impl RendererLeaseError {
    pub fn owner(&self) -> Option<&RendererOwner> {
        match self {
            Self::Busy { owner, .. } => owner.as_ref(),
            Self::Io(_) => None,
        }
    }
}

impl fmt::Display for RendererLeaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy {
                requested,
                owner: Some(owner),
            } => write!(
                formatter,
                "cannot start {requested}: pid {} already owns the GPU renderer as {} ({})",
                owner.pid, owner.role, owner.command
            ),
            Self::Busy {
                requested,
                owner: None,
            } => write!(
                formatter,
                "cannot start {requested}: another process already owns the GPU renderer"
            ),
            Self::Io(error) => write!(formatter, "GPU renderer lease failed: {error}"),
        }
    }
}

impl std::error::Error for RendererLeaseError {}

impl From<io::Error> for RendererLeaseError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Held for the full lifetime of one game or capture-host render process.
///
/// The OS releases the underlying mutex/file lock if the process aborts, which
/// is the property PID files could not provide for the exact OOM failure this
/// guard exists to prevent.
#[derive(Debug)]
pub struct RendererLease {
    owner_path: PathBuf,
    #[cfg(windows)]
    mutex: windows_sys::Win32::Foundation::HANDLE,
    #[cfg(unix)]
    file: fs::File,
}

impl RendererLease {
    /// Acquire the one machine-wide renderer slot without waiting.
    ///
    /// Refusal happens before Bevy creates an adapter/device, so probing a busy
    /// machine cannot itself add GPU pressure.
    pub fn acquire(role: RendererRole) -> Result<Self, RendererLeaseError> {
        let owner_path = std::env::temp_dir().join(OWNER_DIR).join(OWNER_FILE);
        #[cfg(windows)]
        {
            Self::acquire_at(role, owner_path, RENDERER_MUTEX_NAME)
        }
        #[cfg(unix)]
        {
            Self::acquire_at(role, owner_path)
        }
    }

    #[cfg(windows)]
    fn acquire_at(
        role: RendererRole,
        owner_path: PathBuf,
        mutex_name: &str,
    ) -> Result<Self, RendererLeaseError> {
        use windows_sys::Win32::{
            Foundation::{
                CloseHandle, HANDLE, WAIT_ABANDONED, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT,
            },
            System::Threading::{CreateMutexW, ReleaseMutex, WaitForSingleObject},
        };

        if let Some(parent) = owner_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let wide_name = mutex_name
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let mutex: HANDLE = unsafe { CreateMutexW(std::ptr::null(), 0, wide_name.as_ptr()) };
        if mutex.is_null() {
            return Err(io::Error::last_os_error().into());
        }

        match unsafe { WaitForSingleObject(mutex, 0) } {
            WAIT_OBJECT_0 | WAIT_ABANDONED => {
                if let Err(error) = write_owner(&owner_path, &RendererOwner::current(role)) {
                    unsafe {
                        ReleaseMutex(mutex);
                        CloseHandle(mutex);
                    }
                    return Err(error.into());
                }
                Ok(Self { owner_path, mutex })
            }
            WAIT_TIMEOUT => {
                let owner = read_owner(&owner_path);
                unsafe {
                    CloseHandle(mutex);
                }
                Err(RendererLeaseError::Busy {
                    requested: role,
                    owner,
                })
            }
            WAIT_FAILED => {
                let error = io::Error::last_os_error();
                unsafe {
                    CloseHandle(mutex);
                }
                Err(error.into())
            }
            status => {
                unsafe {
                    CloseHandle(mutex);
                }
                Err(io::Error::other(format!(
                    "renderer mutex returned unexpected wait status {status:#x}"
                ))
                .into())
            }
        }
    }

    #[cfg(unix)]
    fn acquire_at(role: RendererRole, owner_path: PathBuf) -> Result<Self, RendererLeaseError> {
        use std::os::fd::AsRawFd;

        if let Some(parent) = owner_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&owner_path)?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            let error = io::Error::last_os_error();
            if matches!(
                error.raw_os_error(),
                Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN
            ) {
                return Err(RendererLeaseError::Busy {
                    requested: role,
                    owner: read_owner(&owner_path),
                });
            }
            return Err(error.into());
        }
        write_owner_to(&mut file, &RendererOwner::current(role))?;
        Ok(Self { owner_path, file })
    }
}

#[cfg(windows)]
impl Drop for RendererLease {
    fn drop(&mut self) {
        use windows_sys::Win32::{Foundation::CloseHandle, System::Threading::ReleaseMutex};

        let _ = fs::remove_file(&self.owner_path);
        unsafe {
            ReleaseMutex(self.mutex);
            CloseHandle(self.mutex);
        }
    }
}

#[cfg(unix)]
impl Drop for RendererLease {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;

        // Clear advisory metadata while still owning the lock. Keep the inode:
        // unlinking a locked file would let a new opener lock a different inode.
        let _ = self.file.set_len(0);
        let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        let _ = &self.owner_path;
    }
}

fn write_owner(path: &Path, owner: &RendererOwner) -> io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    write_owner_to(&mut file, owner)
}

fn write_owner_to(file: &mut fs::File, owner: &RendererOwner) -> io::Result<()> {
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    serde_json::to_writer_pretty(&mut *file, owner).map_err(io::Error::other)?;
    file.write_all(b"\n")?;
    file.sync_all()
}

fn read_owner(path: &Path) -> Option<RendererOwner> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique(tag: &str) -> (PathBuf, String) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "thalos-renderer-lease-test-{tag}-{}-{nonce}",
            std::process::id()
        ));
        (
            dir.join(OWNER_FILE),
            format!(
                "Local\\ThalosRendererLeaseTest{}{}",
                std::process::id(),
                nonce
            ),
        )
    }

    #[test]
    fn role_names_are_stable_metadata() {
        assert_eq!(
            serde_json::to_string(&RendererRole::InteractiveGame).unwrap(),
            "\"interactive-game\""
        );
        assert_eq!(
            serde_json::to_string(&RendererRole::CaptureHost).unwrap(),
            "\"capture-host\""
        );
    }

    #[test]
    fn second_renderer_is_refused_and_release_restores_ownership() {
        let (path, name) = unique("exclusive");
        #[cfg(windows)]
        let first =
            RendererLease::acquire_at(RendererRole::InteractiveGame, path.clone(), &name).unwrap();
        #[cfg(unix)]
        let first = RendererLease::acquire_at(RendererRole::InteractiveGame, path.clone()).unwrap();

        let contender_path = path.clone();
        let contender_name = name.clone();
        let busy = std::thread::spawn(move || {
            #[cfg(windows)]
            let result = {
                RendererLease::acquire_at(
                    RendererRole::CaptureHost,
                    contender_path,
                    &contender_name,
                )
            };
            #[cfg(unix)]
            let result = {
                let _ = contender_name;
                RendererLease::acquire_at(RendererRole::CaptureHost, contender_path)
            };
            match result {
                Ok(_) => panic!("second renderer unexpectedly acquired the lease"),
                Err(error) => error,
            }
        })
        .join()
        .unwrap();

        let owner = busy.owner().expect("busy lease publishes its owner");
        assert_eq!(owner.pid, std::process::id());
        assert_eq!(owner.role, RendererRole::InteractiveGame);
        drop(first);

        #[cfg(windows)]
        let second =
            RendererLease::acquire_at(RendererRole::CaptureHost, path.clone(), &name).unwrap();
        #[cfg(unix)]
        let second = RendererLease::acquire_at(RendererRole::CaptureHost, path.clone()).unwrap();
        drop(second);

        let _ = fs::remove_file(&path);
        if let Some(parent) = path.parent() {
            let _ = fs::remove_dir(parent);
        }
    }
}
