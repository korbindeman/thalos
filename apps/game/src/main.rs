fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("--build-info") => {
            print!("{}", thalos_runtime::distribution::build_info());
            return;
        }
        Some("--verify-install") => match thalos_runtime::distribution::verify_install() {
            Ok(report) => {
                print!("{report}");
                return;
            }
            Err(error) => {
                eprintln!("Thalos install verification failed: {error}");
                std::process::exit(2);
            }
        },
        _ => {}
    }
    thalos_runtime::AppBuilder::new().run();
}
