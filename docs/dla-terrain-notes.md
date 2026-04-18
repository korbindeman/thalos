# DLA terrain generation — optimizations and natural-look tricks

Notes pulled from the transcript. Focus: what makes naive DLA unusable, how to fix it, and how to turn the raw ridge pattern into a believable heightmap.

## The core problem with naive DLA

DLA (diffusion-limited aggregation) works by seeding a pixel, then releasing random walkers one at a time until they collide with the existing cluster and freeze. The resulting sprawl looks like mountain ridgelines, which is why it's interesting for terrain.

It's also unusably slow in its naive form. Early walkers have a million empty cells to wander through before they hit anything, so the first pixels take forever to stick. And the post-processing to turn ridgelines into a heightmap needs very large Gaussian blurs, which are expensive on their own.

## Optimization 1: multi-resolution DLA

Instead of running DLA at full resolution, run it at a small resolution first, upscale, and run again.

- Start with a small image — walkers find the cluster quickly because there's little empty space.
- Once the small image has enough density, upscale it.
- The upscaled image is already well-populated, so new walkers on this larger canvas also find neighbors fast.
- Repeat until you hit the target resolution.

This dodges the "empty canvas" problem at every scale. It also happens to give you a natural fractal structure for free, which feeds into the next trick.

## Optimization 2: crisp upscaling via connection tracking

Plain image upscaling (nearest neighbor, bilinear) destroys the thin ridgeline structure — you can't just blur DLA output and expect it to survive.

The trick is to track *which pixel each new pixel stuck to* during DLA. That gives you a graph of parent-child connections. To upscale, redraw those connections as lines on a larger canvas. The topology is preserved exactly, and you now have room between the lines to add finer DLA detail in the next pass.

Two practical notes:
- Before drawing, split each edge into two segments and jitter the midpoint. Otherwise you get visibly straight lines where the edges used to be.
- This is what makes the multi-resolution approach actually work. Without connection tracking, upscaling would smear the ridges into mush.

## Optimization 3: dual-filter blur instead of a huge Gaussian

The heightmap step needs a very wide blur to spread ridges into mountain shapes. A single large-kernel Gaussian is prohibitive.

Dual-filter blur is a standard trick: downsample repeatedly with a small blur at each step, then upsample repeatedly with a small blur at each step. Small blurs at small resolutions act over a large percentage of the image, so the cumulative effect is a very wide blur for very little work.

Since the multi-resolution DLA pipeline is already producing images at progressively larger sizes, you can piggyback on it:
- Skip the downsample half of dual-filter blur entirely — the low-res DLA image *is* your starting small image.
- Each time you upscale to add more DLA detail, produce two versions: a crisp one (for adding the next round of fine detail) and a blurry one (the accumulating heightmap).
- Feed new detail into both: add it crisply to the crisp image, add it to the blurry image before the next blurry upscale.

Net effect: coarse structure gets blurred many times (becomes broad mountain mass), fine structure gets blurred few times (stays as sharp ridges near the peaks). This is itself a fractal process and runs fast.

## Making it look natural

Two things matter beyond just generating ridges: peak elevation and the transition between coarse and fine scales.

### Weighting ridges so centers are higher than edges

If every ridge pixel has the same value going into the blur, everything ends up at the same elevation — flat-topped. You want peaks in the middle of the web and lower elevations at the extremities.

Use the connection graph:
1. Assign the outermost (leaf) pixels a weight of 1.
2. For every other pixel, set its weight to `max(weights of pixels downstream of it) + 1`.
3. Pixels near the seed end up with the highest weights, leaves stay at 1.

After blurring, this gives you a proper mountain profile rather than a plateau.

### Clamping so fine detail doesn't pile up height

Finer DLA passes keep adding weight on top of already-high central pixels, which makes peaks unrealistically tall and spiky. Fix: apply a smooth falloff so weights asymptote toward a maximum rather than adding linearly.

The transcript reuses the same falloff formula from the gradient-noise trick earlier in the video — the one where influence smoothly saturates as the input grows. Small new contributions near low totals add almost fully; contributions on top of already-large totals barely register. Transitions between detail levels stay gradual and nothing blows up near the summit.

## Chunked generation

Unlike erosion, DLA can be made chunk-friendly with a bit of work. Partition the world into mountainous regions using something like cell/Voronoi noise. For each chunk, check which region it falls in, and run DLA once for the whole region — constrained to only spawn and walk within that region's bounds. Neighboring regions don't interact, so you can generate them independently. This is the same pattern Minecraft uses for multi-chunk structures like villages and dungeons.

## Limitation: no GPU port

DLA is inherently serial. Each new walker can collide with the pixel placed immediately before it, so you can't release many walkers in parallel without changing the algorithm's behavior. The multi-resolution and dual-filter-blur tricks help a lot on CPU, but if you need GPU-resident terrain generation, DLA isn't the tool — the gradient-noise trick from the first half of the video is a better fit there.

## Summary of the pipeline

1. Small canvas, run DLA, track parent-child connections.
2. Upscale by redrawing connections as jittered line segments; this is the crisp image.
3. Run DLA again on the crisp image to add finer detail; update the connection graph.
4. In parallel, maintain a blurry image: blurry-upscale it, then add the new detail.
5. Repeat 2–4 until target resolution.
6. Weight ridge pixels by downstream-depth + 1, with smooth falloff to clamp peaks.
7. The final blurry image is the heightmap.
