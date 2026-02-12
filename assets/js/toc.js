// 目录自动高亮和滚动跟随功能
(function() {
    'use strict';

    // 等待DOM加载完成
    document.addEventListener('DOMContentLoaded', function() {
        const toc = document.querySelector('.toc');
        if (!toc) return;

        const tocLinks = toc.querySelectorAll('a');
        if (tocLinks.length === 0) return;

        // 获取所有标题
        const headings = Array.from(tocLinks).map(link => {
            const id = link.getAttribute('href').substring(1);
            const heading = document.getElementById(id);
            return heading ? { element: heading, link } : null;
        }).filter(item => item !== null);

        if (headings.length === 0) return;

        // 当前激活的索引
        let activeIndex = -1;

        // 滚动时更新激活状态
        function updateActiveHeading() {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const windowHeight = window.innerHeight;

            // 找到当前视口中最近的标题
            let newIndex = -1;

            for (let i = headings.length - 1; i >= 0; i--) {
                const heading = headings[i].element;
                const headingTop = heading.getBoundingClientRect().top + scrollTop;

                // 如果标题在视口上方或正在视口中
                if (headingTop <= scrollTop + windowHeight * 0.3) {
                    newIndex = i;
                    break;
                }
            }

            // 如果找到了新的激活标题
            if (newIndex !== activeIndex) {
                // 移除旧的激活状态
                if (activeIndex !== -1 && headings[activeIndex]) {
                    headings[activeIndex].link.classList.remove('active');
                }

                // 添加新的激活状态
                if (newIndex !== -1 && headings[newIndex]) {
                    headings[newIndex].link.classList.add('active');

                    // 确保激活的链接在可视区域内
                    const linkElement = headings[newIndex].link;
                    const tocInner = toc.querySelector('.inner');

                    if (tocInner) {
                        const tocRect = tocInner.getBoundingClientRect();
                        const linkRect = linkElement.getBoundingClientRect();

                        // 如果链接不在可视区域内，滚动到可视区域
                        if (linkRect.top < tocRect.top || linkRect.bottom > tocRect.bottom) {
                            const scrollPosition = linkElement.offsetTop - tocInner.offsetTop - (tocRect.height / 2) + (linkRect.height / 2);
                            tocInner.scrollTo({
                                top: scrollPosition,
                                behavior: 'smooth'
                            });
                        }
                    }
                }

                activeIndex = newIndex;
            }
        }

        // 使用节流函数优化性能
        let ticking = false;
        function onScroll() {
            if (!ticking) {
                window.requestAnimationFrame(function() {
                    updateActiveHeading();
                    ticking = false;
                });
                ticking = true;
            }
        }

        // 监听滚动事件
        window.addEventListener('scroll', onScroll, { passive: true });

        // 初始化时调用一次
        updateActiveHeading();

        // 平滑滚动到锚点
        tocLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();

                const targetId = this.getAttribute('href').substring(1);
                const targetHeading = document.getElementById(targetId);

                if (targetHeading) {
                    const offsetTop = targetHeading.getBoundingClientRect().top + window.pageYOffset - 80;

                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });

                    // 更新URL但不跳转
                    if (history.pushState) {
                        history.pushState(null, null, '#' + targetId);
                    } else {
                        location.hash = '#' + targetId;
                    }
                }
            });
        });

        // 处理页面加载时的hash
        if (window.location.hash) {
            setTimeout(function() {
                const targetId = window.location.hash.substring(1);
                const targetHeading = document.getElementById(targetId);

                if (targetHeading) {
                    const offsetTop = targetHeading.getBoundingClientRect().top + window.pageYOffset - 80;
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                }
            }, 100);
        }
    });
})();
