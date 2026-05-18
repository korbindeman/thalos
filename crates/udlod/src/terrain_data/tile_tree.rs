use crate::{
    math::{Coordinate, TerrainModel, TileCoordinate},
    terrain_data::{sample_height, tile_atlas::TileAtlas, INVALID_ATLAS_INDEX, INVALID_LOD},
    terrain_view::{TerrainViewComponents, TerrainViewConfig},
    util::inverse_mix,
};
use bevy::{
    math::{DVec2, DVec3},
    platform::collections::HashSet,
    prelude::*,
};
use bytemuck::{Pod, Zeroable};
use itertools::iproduct;
use ndarray::{Array2, Array4};
use std::iter;

/// The current state of a tile of a [`TileTree`].
///
/// This indicates, whether or not the tile should be loaded into the [`TileAtlas`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RequestState {
    /// The tile should be loaded.
    Requested,
    /// The tile does not have to be loaded.
    Released,
}

/// The internal representation of a tile in a [`TileTree`].
struct TileState {
    /// The current tile coordinate at the tile_tree position.
    coordinate: TileCoordinate,
    /// Indicates, whether the tile is currently demanded or released.
    state: RequestState,
}

impl Default for TileState {
    fn default() -> Self {
        Self {
            coordinate: TileCoordinate::INVALID,
            state: RequestState::Released,
        }
    }
}

/// An entry of the [`TileTree`], used to access the best currently loaded tile
/// of the [`TileAtlas`] on the CPU.
///
/// These entries are synced each frame with their equivalent representations in the
/// [`GpuTileTree`](super::gpu_tile_tree::GpuTileTree) for access on the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub(super) struct TileTreeEntry {
    /// The atlas index of the best entry.
    pub(super) atlas_index: u32,
    /// The atlas lod of the best entry.
    pub(super) atlas_lod: u32,
}

impl Default for TileTreeEntry {
    fn default() -> Self {
        Self {
            atlas_index: INVALID_ATLAS_INDEX,
            atlas_lod: INVALID_LOD,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(super) struct TileLookup {
    pub(super) atlas_index: u32,
    pub(super) atlas_lod: u32,
    pub(super) atlas_uv: Vec2,
}

impl TileLookup {
    pub(super) const INVALID: Self = Self {
        atlas_index: INVALID_ATLAS_INDEX,
        atlas_lod: INVALID_LOD,
        atlas_uv: Vec2::ZERO,
    };
}

/// Walks `coord` up the quadtree via [`TileCoordinate::parent`] until its LOD
/// matches `target_lod`. Assumes `target_lod ≤ coord.lod`.
fn ascend_to_lod(coord: TileCoordinate, target_lod: u32) -> TileCoordinate {
    let mut cursor = coord;
    while cursor.lod > target_lod {
        cursor = cursor.parent();
    }
    cursor
}

/// A quadtree-like view of a terrain, that requests and releases tiles from the [`TileAtlas`]
/// depending on the distance to the viewer.
///
/// It can be used to access the best currently loaded tile of the [`TileAtlas`].
/// Additionally its sends this data to the GPU via the
/// [`GpuTileTree`](super::gpu_tile_tree::GpuTileTree) so that it can be utilised
/// in shaders as well.
///
/// Each view (camera, shadow-casting light) that should consider the terrain has to
/// have an associated tile tree.
///
/// This tile tree is a "cube" with a size of (`tree_size`x`tree_size`x`lod_count`), where each layer
/// corresponds to a lod. These layers are wrapping (modulo `tree_size`), that means that
/// the tile tree is always centered under the viewer and only considers `tree_size` / 2 tiles
/// in each direction.
///
/// Each frame the tile tree determines the state of each tile via the
/// `compute_requests` methode.
/// After the [`TileAtlas`] has adjusted to these requests, the tile tree retrieves the best
/// currently loaded tiles from the tile atlas via the `adjust` methode, which can later be used to access the terrain data.
#[derive(Component)]
pub struct TileTree {
    pub(super) origins: Array2<UVec2>,
    /// The current cpu tile_tree data. This is synced each frame with the gpu tile_tree data.
    pub(super) data: Array4<TileTreeEntry>,
    /// Tiles that are no longer required by this tile_tree.
    pub(super) released_tiles: Vec<TileCoordinate>,
    /// Tiles that are requested to be loaded by this tile_tree.
    pub(super) requested_tiles: Vec<TileCoordinate>,
    /// The internal tile states of the tile_tree.
    tiles: Array4<TileState>,
    /// Tiles promoted to "requested" by the 2:1 balance pass that fall outside
    /// the per-side tree window for their LOD. Disjoint from in-window state
    /// in `tiles`: in-window promotions reuse the slot's `Released → Requested`
    /// transition; this set tracks the rest. Delta'd against the previous
    /// frame's set to drive `requested_tiles` / `released_tiles` for the
    /// refcounted atlas — so each coord lands in those vectors at most once
    /// per frame.
    forced_requests: HashSet<TileCoordinate>,
    /// Flat list of tiles to draw this frame, after distance-driven
    /// refinement and 2:1 balance enforcement. Replaces the GPU
    /// `refine_tiles.wgsl` compute pass: the per-tile-independent GPU
    /// predicate cannot enforce gap-≤-1 across the whole frustum, so the
    /// renderer would emit drawn tiles with LOD gap ≥ 2 at face seams and
    /// produce visible elevation shears. The CPU pass runs once per frame
    /// in [`compute_draw_set`] after [`compute_requests`] /
    /// [`adjust_to_tile_atlas`] / [`approximate_height`] have settled, and
    /// the render-world `TerrainViewData::extract` copies this directly
    /// into the `final_tile_buffer` the vertex shader reads.
    pub(crate) draw_set: Vec<TileCoordinate>,
    /// The count of level of detail layers.
    lod_count: u32,
    /// The count of tiles in x and y direction per layer.
    pub(crate) tree_size: u32,
    pub(crate) geometry_tile_count: u32,
    pub(crate) refinement_count: u32,
    pub(crate) grid_size: u32,
    pub(crate) morph_distance: f64,
    pub(crate) blend_distance: f64,
    pub(crate) load_distance: f64,
    pub(crate) subdivision_distance: f64,
    pub(crate) precision_threshold_distance: f64,
    pub(crate) morph_range: f32,
    pub(crate) blend_range: f32,
    pub(crate) origin_lod: u32,
    pub(crate) view_world_position: DVec3,
    pub(crate) approximate_height: f32,
}

impl TileTree {
    /// Creates a new tile_tree from a terrain and a terrain view config.
    pub fn new(tile_atlas: &TileAtlas, view_config: &TerrainViewConfig) -> Self {
        let model = &tile_atlas.model;
        let scale = model.scale();

        Self {
            lod_count: tile_atlas.lod_count,
            tree_size: view_config.tree_size,
            geometry_tile_count: view_config.geometry_tile_count,
            refinement_count: view_config.refinement_count,
            grid_size: view_config.grid_size,
            morph_distance: view_config.morph_distance * scale,
            blend_distance: view_config.blend_distance * scale,
            load_distance: view_config.load_distance * scale,
            subdivision_distance: view_config.morph_distance
                * scale
                * (1.0 + view_config.subdivision_tolerance),
            morph_range: view_config.morph_range,
            blend_range: view_config.blend_range,
            precision_threshold_distance: view_config.precision_threshold_distance * scale,
            origin_lod: view_config.origin_lod,
            view_world_position: default(),
            approximate_height: (model.min_height + model.max_height) / 2.0,
            origins: Array2::default((model.side_count() as usize, tile_atlas.lod_count as usize)),
            data: Array4::default((
                model.side_count() as usize,
                tile_atlas.lod_count as usize,
                view_config.tree_size as usize,
                view_config.tree_size as usize,
            )),
            tiles: Array4::default((
                model.side_count() as usize,
                tile_atlas.lod_count as usize,
                view_config.tree_size as usize,
                view_config.tree_size as usize,
            )),
            released_tiles: default(),
            requested_tiles: default(),
            forced_requests: HashSet::default(),
            draw_set: Vec::new(),
        }
    }

    fn compute_tree_xy(coordinate: Coordinate, tile_count: f64) -> DVec2 {
        // scale and clamp the coordinate to the tile tree bounds
        (coordinate.uv * tile_count).min(DVec2::splat(tile_count - 0.000001))
    }

    fn compute_origin(&self, coordinate: Coordinate, lod: u32) -> UVec2 {
        let tile_count = TileCoordinate::count(lod) as f64;
        let tree_xy = Self::compute_tree_xy(coordinate, tile_count);

        (tree_xy - 0.5 * self.tree_size as f64)
            .round()
            .clamp(
                DVec2::splat(0.0),
                DVec2::splat(tile_count - self.tree_size as f64),
            )
            .as_uvec2()
    }

    fn compute_tile_distance(
        &self,
        tile: TileCoordinate,
        view_coordinate: Coordinate,
        model: &TerrainModel,
    ) -> f64 {
        let tile_count = TileCoordinate::count(tile.lod) as f64;
        let tile_xy = IVec2::new(tile.x as i32, tile.y as i32);
        let view_tile_xy = Self::compute_tree_xy(view_coordinate, tile_count);
        let tile_offset = view_tile_xy.as_ivec2() - tile_xy;
        let mut offset = view_tile_xy % 1.0;

        if tile_offset.x < 0 {
            offset.x = 0.0;
        } else if tile_offset.x > 0 {
            offset.x = 1.0;
        }
        if tile_offset.y < 0 {
            offset.y = 0.0;
        } else if tile_offset.y > 0 {
            offset.y = 1.0;
        }

        let tile_world_position =
            Coordinate::new(tile.side, (tile_xy.as_dvec2() + offset) / tile_count)
                .world_position(model, self.approximate_height);

        tile_world_position.distance(self.view_world_position)
    }

    pub(super) fn compute_blend(&self, sample_world_position: DVec3) -> (u32, f32) {
        let view_distance = self.view_world_position.distance(sample_world_position);
        let target_lod = (self.blend_distance / view_distance)
            .log2()
            .min(self.lod_count as f64 - 0.00001) as f32;
        let lod = target_lod as u32;

        let ratio = if lod == 0 {
            0.0
        } else {
            inverse_mix(lod as f32 + self.blend_range, lod as f32, target_lod)
        };

        (lod, ratio)
    }

    pub(super) fn lookup_tile(
        &self,
        world_position: DVec3,
        tree_lod: u32,
        model: &TerrainModel,
    ) -> TileLookup {
        let coordinate = Coordinate::from_world_position(world_position, model);

        let tile_count = TileCoordinate::count(tree_lod) as f64;
        let tree_xy = Self::compute_tree_xy(coordinate, tile_count);

        let entry = self.data[[
            coordinate.side as usize,
            tree_lod as usize,
            tree_xy.x as usize % self.tree_size as usize,
            tree_xy.y as usize % self.tree_size as usize,
        ]];

        if entry.atlas_lod == INVALID_LOD {
            return TileLookup::INVALID;
        }

        TileLookup {
            atlas_index: entry.atlas_index,
            atlas_lod: entry.atlas_lod,
            atlas_uv: ((tree_xy / (1 << (tree_lod - entry.atlas_lod)) as f64) % 1.0).as_vec2(),
        }
    }

    /// Returns the LOD of the atlas tile the renderer would actually draw at
    /// `world_position` (in the terrain model's local frame). The returned
    /// value is the resident `atlas_lod` — coarser than the deepest tracked
    /// LOD if the fine tile hasn't finished baking yet.
    ///
    /// `None` is returned only when no ancestor tile is resident either
    /// (early frames before any bake has completed).
    ///
    /// Use this to align CPU height queries with what the GPU mesh is
    /// drawing. Passing a hard-coded fine `tile_lod_m` to a CPU height
    /// query while the GPU still shows a coarse parent tile produces a
    /// CPU/GPU height gap — characters appear to float above the visible
    /// terrain by the missing-octave amplitude.
    pub fn best_resident_atlas_lod(
        &self,
        world_position: DVec3,
        model: &TerrainModel,
    ) -> Option<u32> {
        if self.lod_count == 0 {
            return None;
        }
        // The deepest tree_lod is always reachable for a player near the
        // camera (the tile_tree window is centered on the view). The
        // bucket's `atlas_lod` field is already the resident ancestor
        // selected by `adjust_to_tile_atlas`, so a single lookup suffices.
        let lookup = self.lookup_tile(world_position, self.lod_count - 1, model);
        if lookup.atlas_lod == INVALID_LOD {
            None
        } else {
            Some(lookup.atlas_lod)
        }
    }

    fn update(&mut self, view_position: DVec3, tile_atlas: &TileAtlas) {
        let model = &tile_atlas.model;
        self.view_world_position = view_position;

        let view_coordinate = Coordinate::from_world_position(self.view_world_position, model);

        for side in 0..model.side_count() {
            let view_coordinate = view_coordinate.project_to_side(side, model);

            for lod in 0..tile_atlas.lod_count {
                let origin = self.compute_origin(view_coordinate, lod);
                self.origins[(side as usize, lod as usize)] = origin;

                // At low LODs `TileCoordinate::count(lod) < tree_size` (LOD 0
                // has 1×1 tiles, LOD 1 has 2×2, …). The 8×8 sweep below would
                // otherwise produce coordinates beyond the LOD's tile grid;
                // disk providers used to drop those via the `existing_tiles`
                // gate, but synthesised providers need an explicit bound or
                // the atlas overflows on spurious LOD-0 requests.
                let lod_count = TileCoordinate::count(lod);

                for (x, y) in iproduct!(0..self.tree_size, 0..self.tree_size) {
                    let tile_x = origin.x + x;
                    let tile_y = origin.y + y;
                    if tile_x >= lod_count || tile_y >= lod_count {
                        continue;
                    }
                    let tile_coordinate = TileCoordinate {
                        side,
                        lod,
                        x: tile_x,
                        y: tile_y,
                    };

                    let tile_distance =
                        self.compute_tile_distance(tile_coordinate, view_coordinate, model);
                    let load_distance =
                        self.load_distance / TileCoordinate::count(tile_coordinate.lod) as f64;

                    let state = if lod == 0 || tile_distance < load_distance {
                        RequestState::Requested
                    } else {
                        RequestState::Released
                    };

                    let tile = &mut self.tiles[[
                        side as usize,
                        lod as usize,
                        (tile_coordinate.x % self.tree_size) as usize,
                        (tile_coordinate.y % self.tree_size) as usize,
                    ]];

                    // check if tile_tree slot refers to a new tile
                    if tile_coordinate != tile.coordinate {
                        // release old tile
                        if tile.state == RequestState::Requested {
                            tile.state = RequestState::Released;
                            self.released_tiles.push(tile.coordinate);
                        }

                        tile.coordinate = tile_coordinate;
                    }

                    // request or release tile based on its distance to the view
                    match (tile.state, state) {
                        (RequestState::Released, RequestState::Requested) => {
                            tile.state = RequestState::Requested;
                            self.requested_tiles.push(tile.coordinate);
                        }
                        (RequestState::Requested, RequestState::Released) => {
                            tile.state = RequestState::Released;
                            self.released_tiles.push(tile.coordinate);
                        }
                        (_, _) => {}
                    }
                }
            }
        }

        self.balance_lod_gaps(model.is_spherical());
    }

    /// Enforces the 2:1 balanced restricted quadtree constraint across the
    /// request set, including across cube-face boundaries.
    ///
    /// **Invariant**: for every Requested coordinate `T` at LOD `L > 0`, every
    /// non-`INVALID` neighbour (cardinal + corner, possibly on another cube
    /// side) must be covered by some Requested ancestor at LOD `≥ L - 1`.
    ///
    /// Without this, the per-side distance-driven request logic can place
    /// adjacent atlas tiles many LODs apart at cube-face seams (the cross-face
    /// tree window centres on the seam edge, so far-side tiles never reach a
    /// fine LOD). UDLOD's CDLOD morph assumes a ≤ 1 LOD gap, so without
    /// balancing the seam shows as a stair-step elevation crack.
    ///
    /// Promotions for tiles whose tree slot is *in window* re-use the slot's
    /// `Released → Requested` transition. Out-of-window promotions land in
    /// [`Self::forced_requests`]; the delta against the prior frame drives
    /// the refcounted atlas `request_tile` / `release_tile` calls without
    /// double-counting.
    fn balance_lod_gaps(&mut self, spherical: bool) {
        let prev_forced = std::mem::take(&mut self.forced_requests);
        let mut next_forced: HashSet<TileCoordinate> = HashSet::default();

        // Finest → coarsest. Each promotion targets `lod - 1`, which is
        // visited later in the same outer pass, so a single sweep is enough
        // to propagate the stair-step. The `lod_count` outer cap is purely
        // defensive against any future iteration-order regressions.
        let mut iters = 0u32;
        let mut changed = true;
        while changed && iters < self.lod_count {
            changed = false;
            iters += 1;

            for lod in (1..self.lod_count).rev() {
                let seeds = self.collect_requested_at_lod(lod, &next_forced);
                for coord in seeds {
                    for nb in coord.neighbours(spherical) {
                        if nb == TileCoordinate::INVALID {
                            continue;
                        }
                        if self.is_covered(nb, lod - 1, &next_forced) {
                            continue;
                        }
                        let ancestor = ascend_to_lod(nb, lod - 1);
                        if self.try_request_in_window(ancestor) {
                            changed = true;
                        } else if next_forced.insert(ancestor) {
                            changed = true;
                        }
                    }
                }
            }
        }

        // Emit per-frame deltas for the refcounted atlas. Each coord enters
        // either `requested_tiles` or `released_tiles` (or neither) at most
        // once.
        for added in next_forced.difference(&prev_forced) {
            self.requested_tiles.push(*added);
        }
        for removed in prev_forced.difference(&next_forced) {
            self.released_tiles.push(*removed);
        }
        self.forced_requests = next_forced;
    }

    /// Collects every coord at `lod` that is currently a balance-pass seed —
    /// either an in-window slot whose state is `Requested`, or a previously
    /// force-requested coord from `extra`. Returned by value so callers can
    /// mutate `self` while iterating.
    fn collect_requested_at_lod(
        &self,
        lod: u32,
        extra: &HashSet<TileCoordinate>,
    ) -> Vec<TileCoordinate> {
        let lod_idx = lod as usize;
        let mut out = Vec::new();
        for side in 0..self.tiles.shape()[0] {
            for x in 0..self.tree_size as usize {
                for y in 0..self.tree_size as usize {
                    let slot = &self.tiles[[side, lod_idx, x, y]];
                    if slot.state == RequestState::Requested
                        && slot.coordinate.lod == lod
                        && slot.coordinate != TileCoordinate::INVALID
                    {
                        out.push(slot.coordinate);
                    }
                }
            }
        }
        for coord in extra {
            if coord.lod == lod {
                out.push(*coord);
            }
        }
        out
    }

    /// Returns `true` if any ancestor of `coord` at LOD `≥ min_lod` is already
    /// Requested in either the in-window slot array or the `extra`
    /// force-request set. Walks up via `parent()`.
    fn is_covered(
        &self,
        coord: TileCoordinate,
        min_lod: u32,
        extra: &HashSet<TileCoordinate>,
    ) -> bool {
        let mut cursor = coord;
        loop {
            if self.is_slot_requested(cursor) || extra.contains(&cursor) {
                return true;
            }
            if cursor.lod <= min_lod {
                return false;
            }
            cursor = cursor.parent();
        }
    }

    /// Window-membership-aware check: does the slot at `coord`'s tree index
    /// hold this exact coord with state Requested?
    fn is_slot_requested(&self, coord: TileCoordinate) -> bool {
        let slot = &self.tiles[[
            coord.side as usize,
            coord.lod as usize,
            (coord.x % self.tree_size) as usize,
            (coord.y % self.tree_size) as usize,
        ]];
        slot.state == RequestState::Requested && slot.coordinate == coord
    }

    /// Promotes an in-window Released slot to Requested. Returns `true` if the
    /// slot belongs to `coord` and was either already Requested or just
    /// promoted; `false` if the slot holds a different coord (i.e. `coord` is
    /// out of window for this frame).
    fn try_request_in_window(&mut self, coord: TileCoordinate) -> bool {
        let slot = &mut self.tiles[[
            coord.side as usize,
            coord.lod as usize,
            (coord.x % self.tree_size) as usize,
            (coord.y % self.tree_size) as usize,
        ]];
        if slot.coordinate != coord {
            return false;
        }
        if slot.state == RequestState::Released {
            slot.state = RequestState::Requested;
            self.requested_tiles.push(coord);
        }
        true
    }

    /// Traverses all tile_trees and updates the tile states,
    /// while selecting newly requested and released tiles.
    pub(crate) fn compute_requests(
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        tile_atlases: Query<&TileAtlas>,
        frames: crate::big_space::ReferenceFrames,
        view_transforms: Query<crate::big_space::GridTransformReadOnly>,
    ) {
        for (&(terrain, view), tile_tree) in tile_trees.iter_mut() {
            let tile_atlas = tile_atlases.get(terrain).unwrap();
            let view_transform = view_transforms.get(view).unwrap();

            let frame = frames.parent_grid(terrain).unwrap();
            let view_position = view_transform.position_double(frame);

            tile_tree.update(view_position, tile_atlas);
        }
    }

    /// Adjusts all tile_trees to their corresponding tile atlas
    /// by updating the entries with the best available tiles.
    pub(crate) fn adjust_to_tile_atlas(
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        tile_atlases: Query<&TileAtlas>,
    ) {
        for (&(terrain, _view), tile_tree) in tile_trees.iter_mut() {
            let tile_atlas = tile_atlases.get(terrain).unwrap();

            for (tile, entry) in iter::zip(&tile_tree.tiles, &mut tile_tree.data) {
                *entry = tile_atlas.get_best_tile(tile.coordinate);
            }
        }
    }

    pub(crate) fn approximate_height(
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        tile_atlases: Query<&TileAtlas>,
    ) {
        for (&(terrain, _view), tile_tree) in tile_trees.iter_mut() {
            let tile_atlas = tile_atlases.get(terrain).unwrap();

            tile_tree.approximate_height =
                sample_height(tile_tree, tile_atlas, tile_tree.view_world_position);
        }
    }

    /// CPU-side distance-driven refinement of the per-frame draw set with
    /// 2:1 LOD-gap enforcement across cube-face neighbours. Replaces the
    /// GPU `refine_tiles` compute pass.
    ///
    /// The GPU predicate is per-tile-independent (each tile evaluates its
    /// own closest-corner distance), so adjacent tiles can land more than
    /// one LOD apart at face seams — the UDLOD CDLOD morph only spans one
    /// LOD, so a gap ≥ 2 shows up as a hard elevation shear at the seam.
    /// Enforcing balance requires global awareness of the set, which is a
    /// poor fit for compute (set membership + iterative propagation across
    /// neighbours), so we do it on the CPU and upload the result.
    ///
    /// Cost is dominated by the leaf count: at typical play distances the
    /// balanced quadtree has a few hundred leaves, and the balance pass
    /// converges in O(`lod_count`) sweeps. Well under a frame on a single
    /// thread.
    pub fn refine_draw_set_balanced(&self, model: &TerrainModel) -> Vec<TileCoordinate> {
        let view_coordinate = Coordinate::from_world_position(self.view_world_position, model);

        // Step 1: distance-driven refinement — direct port of
        // `refine_tiles.wgsl`'s loop, replacing the GPU `should_be_divided`
        // with the same predicate evaluated in f64 on the CPU. The GPU
        // queue starts at the cube roots, splits each tile whose closest
        // corner is inside the per-LOD subdivision range, and finalizes
        // tiles that don't meet the predicate.
        let mut final_tiles: Vec<TileCoordinate> = Vec::new();
        let mut current: Vec<TileCoordinate> = (0..model.side_count())
            .map(|side| TileCoordinate::new(side, 0, 0, 0))
            .collect();

        for _iter in 0..self.refinement_count {
            if current.is_empty() {
                break;
            }
            let mut next: Vec<TileCoordinate> = Vec::with_capacity(current.len() * 4);
            for tile in current.drain(..) {
                let can_split = tile.lod + 1 < self.lod_count;
                if can_split && self.should_subdivide(tile, view_coordinate, model) {
                    for child in tile.children() {
                        next.push(child);
                    }
                } else {
                    final_tiles.push(tile);
                }
            }
            current = next;
        }
        // Anything still pending after the iteration cap drops in at its
        // current LOD — same termination semantics as `refine_tiles.wgsl`,
        // where any tile that didn't get finalized within `refinement_count`
        // iterations remains as a leaf at the maximum reached depth.
        final_tiles.extend(current);

        // Step 2: enforce the 2:1 invariant.
        self.balance_draw_set(&mut final_tiles, model.is_spherical());

        final_tiles
    }

    /// CPU mirror of `refine_tiles.wgsl::should_be_divided`: returns true
    /// iff the distance from the view to the tile's closest corner is less
    /// than `subdivision_distance / 2^lod`. Uses the same closest-corner
    /// logic as [`Self::compute_tile_distance`] (which the streaming pass
    /// also relies on), so streaming and drawing share one distance model.
    fn should_subdivide(
        &self,
        tile: TileCoordinate,
        view_coordinate: Coordinate,
        model: &TerrainModel,
    ) -> bool {
        let projected_view = view_coordinate.project_to_side(tile.side, model);
        let distance = self.compute_tile_distance(tile, projected_view, model);
        let threshold = self.subdivision_distance / TileCoordinate::count(tile.lod) as f64;
        distance < threshold
    }

    /// Iteratively splits any drawn tile whose neighbouring leaf is more
    /// than one LOD coarser. The neighbour search walks `parent()` from the
    /// same-LOD coord; for a balanced quadtree the first ancestor in the
    /// set is the leaf that actually borders this tile. When a gap ≥ 2 is
    /// detected the coarser ancestor is split into its four children and
    /// the loop re-runs — splitting a coarse tile can produce children
    /// that themselves need splitting against a still-finer neighbour, so
    /// the propagation is unbounded a priori. In practice it converges in
    /// at most `lod_count` sweeps because each sweep advances the imbalance
    /// frontier one LOD step.
    fn balance_draw_set(&self, tiles: &mut Vec<TileCoordinate>, spherical: bool) {
        let mut set: HashSet<TileCoordinate> = tiles.drain(..).collect();
        let max_iters = self.lod_count.saturating_add(2);
        let mut changed = true;
        let mut iters = 0u32;
        while changed && iters < max_iters {
            changed = false;
            iters += 1;
            // Snapshot before mutating: splitting `cursor` invalidates the
            // iterator's view of `set` and we'd skip newly-added tiles
            // until the next sweep anyway. The outer loop catches them.
            let snapshot: Vec<TileCoordinate> = set.iter().copied().collect();
            for tile in snapshot {
                if !set.contains(&tile) {
                    // Already split out from under us by a prior step in
                    // this sweep — skip.
                    continue;
                }
                if tile.lod <= 1 {
                    // A tile at LOD 0 or 1 cannot have a neighbour more
                    // than one LOD coarser (neighbours at LOD 0 can only
                    // be at LOD 0). Cheap skip.
                    continue;
                }
                for nb in tile.neighbours(spherical) {
                    if nb == TileCoordinate::INVALID {
                        continue;
                    }
                    let mut cursor = nb;
                    while !set.contains(&cursor) {
                        if cursor.lod == 0 {
                            break;
                        }
                        cursor = cursor.parent();
                    }
                    if !set.contains(&cursor) {
                        // No ancestor of the neighbour is in the draw set.
                        // This can only happen at startup before LOD 0 has
                        // been added on a side the camera can't see — in
                        // that case the absent tile simply doesn't draw,
                        // and there's no seam to fix.
                        continue;
                    }
                    if cursor.lod + 1 < tile.lod {
                        set.remove(&cursor);
                        for child in cursor.children() {
                            set.insert(child);
                        }
                        changed = true;
                    }
                }
            }
        }
        tiles.extend(set);
    }

    /// Recomputes the draw set for every (terrain, view) pair after
    /// streaming has settled for the frame. Runs once per frame in
    /// `Last` after [`Self::approximate_height`] so the refinement sees a
    /// fresh `view_world_position` and the current best-resident height
    /// estimate (used by `compute_tile_distance`).
    pub(crate) fn compute_draw_set(
        mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
        tile_atlases: Query<&TileAtlas>,
    ) {
        for (&(terrain, _view), tile_tree) in tile_trees.iter_mut() {
            let Ok(tile_atlas) = tile_atlases.get(terrain) else {
                continue;
            };
            tile_tree.draw_set = tile_tree.refine_draw_set_balanced(&tile_atlas.model);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructs a minimally-populated [`TileTree`] for balance tests. Only
    /// the fields touched by [`TileTree::balance_lod_gaps`] are meaningful;
    /// distance / morph / blend configuration is irrelevant (the balance
    /// pass reads neither the view position nor the load thresholds).
    fn test_tree(lod_count: u32, tree_size: u32) -> TileTree {
        let side_count = 6usize;
        TileTree {
            origins: Array2::default((side_count, lod_count as usize)),
            data: Array4::default((
                side_count,
                lod_count as usize,
                tree_size as usize,
                tree_size as usize,
            )),
            released_tiles: Vec::new(),
            requested_tiles: Vec::new(),
            tiles: Array4::default((
                side_count,
                lod_count as usize,
                tree_size as usize,
                tree_size as usize,
            )),
            forced_requests: HashSet::default(),
            draw_set: Vec::new(),
            lod_count,
            tree_size,
            geometry_tile_count: 0,
            refinement_count: 0,
            grid_size: 0,
            morph_distance: 0.0,
            blend_distance: 0.0,
            load_distance: 0.0,
            subdivision_distance: 0.0,
            precision_threshold_distance: 0.0,
            morph_range: 0.0,
            blend_range: 0.0,
            origin_lod: 0,
            view_world_position: DVec3::ZERO,
            approximate_height: 0.0,
        }
    }

    /// Seeds a Requested tile at the in-window slot for `coord`. Assumes the
    /// caller picked `coord` so its xy fits inside the tree window (i.e.
    /// `coord.x % tree_size == coord.x`, which holds for small lods or for
    /// xy values < tree_size).
    fn seed_requested(tree: &mut TileTree, coord: TileCoordinate) {
        let slot = &mut tree.tiles[[
            coord.side as usize,
            coord.lod as usize,
            (coord.x % tree.tree_size) as usize,
            (coord.y % tree.tree_size) as usize,
        ]];
        slot.coordinate = coord;
        slot.state = RequestState::Requested;
    }

    /// Brute-force coverage walk independent of `is_covered`: returns true
    /// iff some ancestor of `coord` at LOD ≥ `min_lod` is Requested somewhere
    /// in the tree (in-window slot or `forced_requests`).
    fn covered(tree: &TileTree, coord: TileCoordinate, min_lod: u32) -> bool {
        let mut cursor = coord;
        loop {
            let slot = &tree.tiles[[
                cursor.side as usize,
                cursor.lod as usize,
                (cursor.x % tree.tree_size) as usize,
                (cursor.y % tree.tree_size) as usize,
            ]];
            if slot.state == RequestState::Requested && slot.coordinate == cursor {
                return true;
            }
            if tree.forced_requests.contains(&cursor) {
                return true;
            }
            if cursor.lod <= min_lod {
                return false;
            }
            cursor = cursor.parent();
        }
    }

    #[test]
    fn balance_invariant_holds_after_single_face_seed() {
        // tree_size large enough that a LOD-3 tile in the face interior is
        // entirely in-window. Seed one Requested tile in the middle of side 0.
        let mut tree = test_tree(/* lod_count */ 6, /* tree_size */ 16);
        let seed = TileCoordinate::new(0, 4, 8, 8);
        seed_requested(&mut tree, seed);

        tree.balance_lod_gaps(true);

        // Walk every Requested tile and check the invariant by brute force.
        for side in 0..6u32 {
            for lod in 1..tree.lod_count {
                for x in 0..tree.tree_size {
                    for y in 0..tree.tree_size {
                        let slot =
                            &tree.tiles[[side as usize, lod as usize, x as usize, y as usize]];
                        if slot.state != RequestState::Requested {
                            continue;
                        }
                        let coord = slot.coordinate;
                        if coord == TileCoordinate::INVALID {
                            continue;
                        }
                        for nb in coord.neighbours(true) {
                            if nb == TileCoordinate::INVALID {
                                continue;
                            }
                            assert!(
                                covered(&tree, nb, lod - 1),
                                "neighbour {nb:?} of {coord:?} not covered at lod {}",
                                lod - 1,
                            );
                        }
                    }
                }
            }
            for forced in tree.forced_requests.iter() {
                if forced.lod == 0 {
                    continue;
                }
                for nb in forced.neighbours(true) {
                    if nb == TileCoordinate::INVALID {
                        continue;
                    }
                    assert!(
                        covered(&tree, nb, forced.lod - 1),
                        "neighbour {nb:?} of forced {forced:?} not covered at lod {}",
                        forced.lod - 1,
                    );
                }
            }
        }
    }

    /// The draw-set 2:1 invariant: for every drawn tile, every same-LOD
    /// neighbour (or its ancestor that's actually drawn) is at LOD `>= L-1`.
    /// Worst-case stress: one face has a deep leaf in its NE corner while
    /// every other face is still at LOD 0. `balance_draw_set` must
    /// propagate splits across the cube-face seam until no gap ≥ 2
    /// remains. `balance_draw_set` operates purely on set membership, so
    /// the input doesn't have to be a valid quadtree partition.
    #[test]
    fn draw_set_balance_invariant_holds() {
        let tree = test_tree(/* lod_count */ 8, /* tree_size */ 16);

        let deep_lod = 5u32;
        let max_xy = TileCoordinate::count(deep_lod);
        let mut tiles: Vec<TileCoordinate> =
            vec![TileCoordinate::new(0, deep_lod, max_xy - 1, max_xy - 1)];
        for side in 0..6u32 {
            tiles.push(TileCoordinate::new(side, 0, 0, 0));
        }

        tree.balance_draw_set(&mut tiles, /* spherical */ true);

        let set: HashSet<TileCoordinate> = tiles.iter().copied().collect();
        for &tile in &tiles {
            if tile.lod <= 1 {
                continue;
            }
            for nb in tile.neighbours(true) {
                if nb == TileCoordinate::INVALID {
                    continue;
                }
                let mut cursor = nb;
                while !set.contains(&cursor) {
                    if cursor.lod == 0 {
                        break;
                    }
                    cursor = cursor.parent();
                }
                if !set.contains(&cursor) {
                    // Neighbour has no covering tile in the input — not
                    // a balance violation, just an incomplete partition.
                    continue;
                }
                assert!(
                    cursor.lod + 1 >= tile.lod,
                    "draw-set 2:1 violation: tile {tile:?} at lod {} \
                     has neighbour ancestor {cursor:?} at lod {}",
                    tile.lod,
                    cursor.lod,
                );
            }
        }
    }

    #[test]
    fn balance_propagates_across_face_boundary() {
        let mut tree = test_tree(/* lod_count */ 8, /* tree_size */ 16);

        // A tile at the maximum-x boundary of side 0 has a right-cardinal
        // neighbour on a different cube side. Pick `lod=6` so the boundary
        // tile coord is `count(6) - 1 = 63`, and side 0's right-cardinal
        // neighbour is some other side per `NEIGHBOURING_SIDES[0]`.
        let lod = 6u32;
        let max_x = TileCoordinate::count(lod) - 1;
        // pick a y inside the window for an easy seed slot. y in tree_size.
        let seed = TileCoordinate::new(0, lod, max_x, 4);
        seed_requested(&mut tree, seed);

        // The neighbour structure must include at least one cross-face entry
        // for the seed (right-cardinal — index 1 in `OFFSETS`).
        let right_neighbour = seed.neighbours(true).nth(1).unwrap();
        assert_ne!(right_neighbour, TileCoordinate::INVALID);
        assert_ne!(
            right_neighbour.side, seed.side,
            "expected cross-face right neighbour, got same-side {right_neighbour:?}"
        );

        tree.balance_lod_gaps(true);

        // The right neighbour's parent at `lod - 1` is the ancestor that
        // satisfies the invariant. It must now be requested somewhere —
        // either in-window on the cross-face side, or in `forced_requests`.
        let parent = right_neighbour.parent();
        assert_eq!(parent.lod, lod - 1);
        let in_window = {
            let slot = &tree.tiles[[
                parent.side as usize,
                parent.lod as usize,
                (parent.x % tree.tree_size) as usize,
                (parent.y % tree.tree_size) as usize,
            ]];
            slot.state == RequestState::Requested && slot.coordinate == parent
        };
        let forced = tree.forced_requests.contains(&parent);
        assert!(
            in_window || forced,
            "cross-face parent {parent:?} not requested anywhere (in_window={in_window}, forced={forced})"
        );
    }
}
