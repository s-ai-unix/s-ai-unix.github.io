// 将mermaid代码块转换为可渲染的div
document.addEventListener('DOMContentLoaded', function() {
  // 查找所有代码块
  const codeBlocks = document.querySelectorAll('pre > code');

  codeBlocks.forEach((block) => {
    const code = block.textContent.trim();
    const languageClass = block.className.match(/language-(\w+)/);
    const language = languageClass ? languageClass[1] : '';

    // 检查是否是mermaid代码（通过语言标记或内容）
    if (language === 'mermaid' ||
        code.startsWith('flowchart') ||
        code.startsWith('graph') ||
        code.startsWith('gitgraph') ||
        code.startsWith('graph TD') ||
        code.startsWith('graph LR')) {

      // 创建mermaid div
      const mermaidDiv = document.createElement('div');
      mermaidDiv.className = 'mermaid';
      mermaidDiv.textContent = code;

      // 创建白色包装器，用于覆盖父元素的黑色背景
      const wrapper = document.createElement('div');
      wrapper.className = 'mermaid-wrapper';
      wrapper.style.background = '#ffffff';
      wrapper.style.padding = '2rem 1rem';
      wrapper.style.margin = '2rem 0';
      wrapper.style.width = '100%';
      wrapper.style.borderRadius = '8px';
      wrapper.style.boxShadow = '0 2px 12px rgba(0,0,0,0.08)';
      wrapper.appendChild(mermaidDiv);

      // 找到最外层的容器（.highlight 或 .chroma）
      let container = block.parentElement;
      while (container && !container.classList.contains('highlight') && !container.classList.contains('chroma')) {
        container = container.parentElement;
      }

      // 如果找到了容器，替换整个容器；否则只替换 pre
      if (container && (container.classList.contains('highlight') || container.classList.contains('chroma'))) {
        container.replaceWith(wrapper);
      } else {
        const pre = block.parentElement;
        pre.replaceWith(wrapper);
      }
    }
  });
});
