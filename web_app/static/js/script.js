// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 导航栏激活状态
    setActiveNavItem();

    // 图片加载错误处理
    setupImageErrorHandling();

    // 平滑滚动
    setupSmoothScroll();

    // 图片预览功能
    setupImagePreview();
});

// 设置当前页面导航项激活状态
function setActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');

    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// 图片加载错误处理
function setupImageErrorHandling() {
    const images = document.querySelectorAll('img');

    images.forEach(img => {
        img.addEventListener('error', function() {
            this.src = '/static/images/placeholder.png';
            this.alt = '图片加载失败';
        });
    });
}

// 平滑滚动
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();

            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}

// 图片预览功能
function setupImagePreview() {
    const images = document.querySelectorAll('.image-container img');

    images.forEach(img => {
        img.addEventListener('click', function() {
            // 创建预览模态框
            const modal = document.createElement('div');
            modal.style.position = 'fixed';
            modal.style.top = '0';
            modal.style.left = '0';
            modal.style.width = '100%';
            modal.style.height = '100%';
            modal.style.backgroundColor = 'rgba(0,0,0,0.8)';
            modal.style.display = 'flex';
            modal.style.alignItems = 'center';
            modal.style.justifyContent = 'center';
            modal.style.zIndex = '1000';
            modal.style.cursor = 'zoom-out';

            // 创建预览图片
            const previewImg = document.createElement('img');
            previewImg.src = this.src;
            previewImg.style.maxWidth = '90%';
            previewImg.style.maxHeight = '90%';
            previewImg.style.objectFit = 'contain';

            modal.appendChild(previewImg);
            document.body.appendChild(modal);

            // 点击关闭预览
            modal.addEventListener('click', function() {
                document.body.removeChild(modal);
            });
        });
    });
}