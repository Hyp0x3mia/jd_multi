"""
静态文件工具模块，用于创建和管理静态文件
"""

import os
import time

def create_static_files(base_path="oxygent/chart"):
    """创建必要的静态文件"""
    # 创建 index.html
    index_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mermaid 流程图生成器</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="container">
        <h1>Mermaid 流程图生成器</h1>
        
        <div class="input-container">
            <h2>输入流程图描述</h2>
            <textarea id="description" placeholder="请输入流程图描述，例如：生成一个软件开发流程图，包括需求分析、设计、编码、测试和部署阶段。"></textarea>
            <button id="generate-btn">生成流程图</button>
        </div>
        
        <div class="result-container" id="result">
            <!-- 结果将在这里显示 -->
        </div>
    </div>
    
    <script src="js/app.js"></script>
</body>
</html>"""
    
    # 创建 style.css
    style_css = """body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

h1, h2 {
    color: #333;
}

.input-container {
    margin-bottom: 20px;
}

textarea {
    width: 100%;
    height: 150px;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    margin-bottom: 10px;
}

button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
}

button:hover {
    background-color: #45a049;
}

.result-container {
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: #f9f9f9;
    min-height: 100px;
}"""
    
    # 创建 app.js
    app_js = """document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generate-btn');
    const descriptionInput = document.getElementById('description');
    const resultContainer = document.getElementById('result');
    
    generateBtn.addEventListener('click', function() {
        const description = descriptionInput.value.trim();
        
        if (!description) {
            alert('请输入流程图描述');
            return;
        }
        
        // 显示加载状态
        resultContainer.innerHTML = '<p>正在生成流程图，请稍候...</p>';
        
        // 发送请求到后端
        fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ description: description }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                resultContainer.innerHTML = `
                    <p>流程图已生成！</p>
                    <p>文件路径: ${data.file_path}</p>
                    <p>正在打开浏览器...</p>
                `;
            } else {
                resultContainer.innerHTML = `<p>生成失败: ${data.error}</p>`;
            }
        })
        .catch(error => {
            console.error('请求出错:', error);
            resultContainer.innerHTML = '<p>请求出错，请查看控制台获取详细信息</p>';
        });
    });
});"""
    
    # 写入文件
    with open(f"{base_path}/web/index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    with open(f"{base_path}/web/css/style.css", "w", encoding="utf-8") as f:
        f.write(style_css)
    
    with open(f"{base_path}/web/js/app.js", "w", encoding="utf-8") as f:
        f.write(app_js)