# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added support for dynamic registration of `oxy`

---
## [1.0.6.3] - 2025-10-15

### Added
- Added fine-grained message storage, refer to [./examples/advanced/demo_save_message.py](./examples/advanced/demo_save_message.py)

### Changed
- Updated examples. For details, see [./examples](./examples)

---

## [1.0.6.2] - 2025-10-09

### Added
- Added an example of a custom agent input schema, refer to [./examples/advanced/demo_custom_agent_input_schema.py](./examples/advanced/demo_custom_agent_input_schema.py)

### Changed
- Renamed Vearch configuration parameter `tool_df_space_name` to `tool_space_name`
- Modified the names of environment variables used in `config.json`

---

## [1.0.6.1] - 2025-09-30

### Added
- Added multimodal information transfer mechanism between agents, refer to [./examples/advanced/demo_multimodal_transfer.py](./examples/advanced/demo_multimodal_transfer.py)

### Changed
- Automatically generate externally accessible Web links after uploading attachments

### Fixed
- Fixed the issue where multiple interactions between agents in a single conversation were not correctly recorded in history

### Removed
- Removed support for `web_file_url_list` in `payload`
