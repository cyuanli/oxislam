# oxislam

A Rust library for computer vision and SLAM (Simultaneous Localization and Mapping).

## Status

### Implemented
- [x] Image I/O and basic types (Gray, RGB)
- [x] Filters (Gaussian, Sobel)
- [x] Harris corner detector
- [x] FAST corner detector
- [x] ORB detector (multi-scale FAST + orientation)
- [x] Image pyramid
- [x] BRIEF binary descriptor (rotation-aware)
- [x] Patch-based descriptor
- [x] Feature matching (brute-force, ratio test)
- [x] Parallel processing utilities
- [x] Visualization (drawing primitives, canvas composition)

### Planned
- [ ] Additional detectors (SIFT)
- [ ] Pose estimation / essential matrix
- [ ] Bundle adjustment
- [ ] Map/keyframe management
- [ ] Loop closure detection

**Note**: Early stage. API is unstable.

## Features

- **Image Processing**: Filtering (separable Gaussian, Sobel), bilinear resize, image pyramids, pixel types, parallel operations, [`image`](https://crates.io/crates/image) crate interop (behind `image` feature)
- **Geometry**: 2D/3D point and vector types (via nalgebra)
- **Feature Detection**: Harris corner detector, FAST corner detector, ORB detector
- **Feature Description**: BRIEF binary descriptors, patch descriptors
- **Feature Matching**: Brute-force matching with Hamming/L2 distance and ratio test
- **Visualization**: Drawing primitives (cross markers, lines), canvas utilities (side-by-side composition), color helpers

## Quick Start

### Detect Keypoints

See `crates/oxislam-features/examples/detect_keypoints.rs`:

```bash
cargo run --example detect_keypoints -- path/to/image.jpg
```

Detects corners in an image and saves an annotated version.

### Match Features

See `crates/oxislam-features/examples/match_features.rs`:

```bash
cargo run --example match_features -- image1.jpg image2.jpg
```

Detects keypoints, extracts descriptors, and matches features between two images. Outputs a side-by-side visualization with match lines. Supports `--detector` (fast, harris) and `--descriptor` (brief128, brief256, brief512, patch) options.

### Match Video

See `crates/oxislam-features/examples/match_video.rs`:

```bash
cargo run --features rayon --example match_video -- input.mp4 -o output.mp4
```

Demonstrates ORB feature matching stability across video frames. Produces an output video with the first frame on the left and the current frame on the right, with colored match lines drawn between them. Use `--trace trace.json` to emit a Chrome trace file for profiling. Requires `ffmpeg` installed on the system.

## Crates

- **oxislam-image**: Image processing and filtering (interop with the `image` crate behind the `image` feature)
- **oxislam-geometry**: Geometric types
- **oxislam-features**: Feature detection and description (`tracing` feature for span instrumentation)
- **oxislam-viz**: Visualization utilities (drawing, canvas, colors)
