# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-04-22

### Added
- Integrated YOLOv8 for intelligent grain detection and classification.
- Implemented real-time WebRTC camera processing.
- Added Plotly-powered analytical dashboard for grain distribution insights.
- Implemented CSV export functionality for detection results.

### Changed
- Migrated default engine to High-Precision OpenCV Watershed for exact counts.
- Optimized UI aesthetics with Glassmorphism and Outfit typography.
- Refined sidebar navigation for improved user experience.

### Fixed
- Resolved latency issues in real-time camera overlays.
- Fixed layout responsiveness on ultra-wide monitors.

## [1.3.0] - Planned

### Added
- Docker containerization for standardized deployment.
- Initialized dedicated image processing utility module (`src/utils.py`).
- Added CITING.md for standardized research citations.
- Introduced real-time system status indicators in the UI.

### Changed
- Refactored core AI engine to use centralized preprocessing utilities.
- Enhanced sidebar UI with emoji iconography and descriptive tips.
- Optimized main title with custom glow effects and premium aesthetics.

### Fixed
- Corrected metadata typos in the analytical pipeline.
- Resolved type hinting gaps in the training workflow.
