# Changelog

所有重要变更将在此文件中记录。

## [Unreleased]

### Added
- 新增支持动态注册 `oxy`

---

## [1.0.6.3] - 2025-10-15

### Added
- 新增细粒度的消息存储，详见 [./examples/advanced/demo_save_message.py](./examples/advanced/demo_save_message.py)

### Changed
- 更新示例，详见 [./examples](./examples)

---

## [1.0.6.2] - 2025-10-09

### Added
- 新增自定义智能体输入结构体示例，详见 [./examples/advanced/demo_custom_agent_input_schema.py](./examples/advanced/demo_custom_agent_input_schema.py)

### Changed
- Vearch 配置参数 `tool_df_space_name` 更名为 `tool_space_name`
- 修改 `config.json` 中引用的环境变量名称

---

## [1.0.6.1] - 2025-09-30

### Added
- 新增智能体之间的多模态信息传递机制，详见 [./examples/advanced/demo_multimodal_transfer.py](./examples/advanced/demo_multimodal_transfer.py)

### Changed
- 上传附件后，自动生成外部可访问的 Web 链接

### Fixed
- 修复一轮对话中，多智能体之间多次交互未正确记录历史的问题

### Removed
- 移除对 `payload` 中 `web_file_url_list` 的支持
