// 将mermaid代码块转换为可渲染的div
document.addEventListener('DOMContentLoaded', function() {
  // 查找所有代码块
  const codeBlocks = document.querySelectorAll('pre > code');

  codeBlocks.forEach((block) => {
    const code = block.textContent.trim();

    // 检查是否是mermaid代码
    if (code.startsWith('flowchart') || code.startsWith('graph') || code.startsWith('gitgraph')) {
      // 创建mermaid div
      const mermaidDiv = document.createElement('div');
      mermaidDiv.className = 'mermaid';
      mermaidDiv.textContent = code;

      // 替换原代码块
      const pre = block.parentElement;
      pre.replaceWith(mermaidDiv);
    }
  });
});
