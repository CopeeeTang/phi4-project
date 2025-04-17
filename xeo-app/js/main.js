// 主要JavaScript文件

document.addEventListener('DOMContentLoaded', function() {
    // 初始化设备卡片交互
    initDeviceCards();
    
    // 初始化设置滑块
    initSliders();
    
    // 添加粒子背景效果
    initParticleBackground();
    
    // 初始化设备连接状态模拟
    simulateDeviceConnections();
});

// 初始化设备卡片交互
function initDeviceCards() {
    const deviceCards = document.querySelectorAll('.device-card');
    
    deviceCards.forEach(card => {
        card.addEventListener('click', function() {
            // 获取设备信息
            const deviceName = this.querySelector('p').textContent;
            
            // 显示设备控制模态框
            showDeviceControlModal(deviceName, this);
        });
    });
}

// 显示设备控制模态框
function showDeviceControlModal(deviceName, cardElement) {
    // 创建模态框元素
    const modal = document.createElement('div');
    modal.className = 'device-modal';
    
    // 获取设备类型图标
    const deviceIcon = cardElement.querySelector('i').className;
    
    // 创建模态框内容
    modal.innerHTML = `
        <div class="device-modal-content">
            <div class="modal-header">
                <h3>${deviceName}</h3>
                <span class="close-modal">&times;</span>
            </div>
            <div class="modal-body">
                <div class="device-icon">
                    <i class="${deviceIcon}"></i>
                </div>
                <div class="device-controls">
                    <div class="control-group">
                        <button class="control-btn power-btn">
                            <i class="fas fa-power-off"></i> 开/关
                        </button>
                        <button class="control-btn">
                            <i class="fas fa-sliders-h"></i> 控制
                        </button>
                        <button class="control-btn">
                            <i class="fas fa-cog"></i> 设置
                        </button>
                    </div>
                    ${getDeviceSpecificControls(deviceName)}
                </div>
            </div>
        </div>
    `;
    
    // 添加模态框到body
    document.body.appendChild(modal);
    
    // 显示模态框并添加淡入效果
    setTimeout(() => {
        modal.style.opacity = '1';
    }, 10);
    
    // 添加关闭按钮事件监听
    const closeBtn = modal.querySelector('.close-modal');
    closeBtn.addEventListener('click', () => {
        closeModal(modal);
    });
    
    // 点击模态框外部关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal(modal);
        }
    });
    
    // 添加电源按钮交互
    const powerBtn = modal.querySelector('.power-btn');
    powerBtn.addEventListener('click', () => {
        powerBtn.classList.toggle('active');
        // 这里添加设备开关控制代码
        console.log(`${deviceName} 电源状态已切换`);
    });
    
    // 阻止事件冒泡
    modal.querySelector('.device-modal-content').addEventListener('click', (e) => {
        e.stopPropagation();
    });
}

// 关闭模态框
function closeModal(modal) {
    modal.style.opacity = '0';
    setTimeout(() => {
        document.body.removeChild(modal);
    }, 300);
}

// 根据设备类型获取特定控制选项
function getDeviceSpecificControls(deviceName) {
    // 根据设备名称返回对应的控制元素
    if (deviceName.includes('灯')) {
        return `
            <div class="specific-controls">
                <div class="color-picker-container">
                    <p>颜色</p>
                    <input type="color" class="color-picker" value="#3498db">
                </div>
                <div class="brightness-container">
                    <p>亮度</p>
                    <input type="range" min="1" max="100" value="70" class="slider brightness-slider">
                    <span class="slider-value">70%</span>
                </div>
            </div>
        `;
    } else if (deviceName.includes('空调') || deviceName.includes('温度')) {
        return `
            <div class="specific-controls">
                <div class="temperature-container">
                    <p>温度</p>
                    <div class="temp-control">
                        <button class="temp-btn temp-down">-</button>
                        <span class="temp-value">23°C</span>
                        <button class="temp-btn temp-up">+</button>
                    </div>
                </div>
                <div class="mode-container">
                    <p>模式</p>
                    <div class="mode-buttons">
                        <button class="mode-btn active">自动</button>
                        <button class="mode-btn">制冷</button>
                        <button class="mode-btn">制热</button>
                        <button class="mode-btn">风扇</button>
                    </div>
                </div>
            </div>
        `;
    } else if (deviceName.includes('电视') || deviceName.includes('Apple TV')) {
        return `
            <div class="specific-controls">
                <div class="remote-container">
                    <div class="remote-nav">
                        <button class="remote-btn up"><i class="fas fa-chevron-up"></i></button>
                        <button class="remote-btn right"><i class="fas fa-chevron-right"></i></button>
                        <button class="remote-btn down"><i class="fas fa-chevron-down"></i></button>
                        <button class="remote-btn left"><i class="fas fa-chevron-left"></i></button>
                        <button class="remote-btn center">OK</button>
                    </div>
                    <div class="volume-container">
                        <p>音量</p>
                        <input type="range" min="0" max="100" value="50" class="slider volume-slider">
                        <span class="slider-value">50%</span>
                    </div>
                </div>
            </div>
        `;
    } else {
        // 默认控制选项
        return `
            <div class="specific-controls">
                <p>设备状态: <span class="device-status">正常</span></p>
                <div class="status-container">
                    <p>电池电量</p>
                    <div class="battery-indicator">
                        <div class="battery-level" style="width: 65%"></div>
                    </div>
                    <span class="battery-percentage">65%</span>
                </div>
            </div>
        `;
    }
}

// 初始化设置滑块
function initSliders() {
    const sliders = document.querySelectorAll('.slider');
    
    sliders.forEach(slider => {
        const valueDisplay = slider.nextElementSibling;
        
        // 设置初始值
        valueDisplay.textContent = `${slider.value}%`;
        
        // 添加滑块事件监听
        slider.addEventListener('input', function() {
            valueDisplay.textContent = `${this.value}%`;
            
            // 如果是亮度滑块，更新背景颜色
            if (this.classList.contains('brightness-slider')) {
                updateBrightness(this.value);
            }
        });
    });
}

// 更新亮度值
function updateBrightness(value) {
    // 这里可以添加实际的亮度控制代码
    console.log(`亮度已更新为: ${value}%`);
}

// 初始化粒子背景
function initParticleBackground() {
    // 创建粒子背景
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles-container';
    document.querySelector('.container').appendChild(particlesContainer);
    
    // 添加粒子
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // 随机位置和大小
        const size = Math.random() * 3 + 1;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}vw`;
        particle.style.top = `${Math.random() * 100}vh`;
        
        // 随机动画持续时间
        const duration = Math.random() * 50 + 30;
        particle.style.animation = `float ${duration}s linear infinite`;
        particle.style.animationDelay = `${Math.random() * 10}s`;
        
        particlesContainer.appendChild(particle);
    }
    
    // 添加粒子动画样式
    if (!document.getElementById('particle-styles')) {
        const style = document.createElement('style');
        style.id = 'particle-styles';
        style.textContent = `
            .particles-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: -1;
            }
            
            .particle {
                position: absolute;
                background-color: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                pointer-events: none;
            }
            
            @keyframes float {
                0% {
                    transform: translateY(0) translateX(0);
                    opacity: 0;
                }
                10% {
                    opacity: 0.3;
                }
                90% {
                    opacity: 0.3;
                }
                100% {
                    transform: translateY(-100vh) translateX(20px);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// 模拟设备连接状态
function simulateDeviceConnections() {
    const deviceCards = document.querySelectorAll('.device-card');
    
    deviceCards.forEach(card => {
        // 随机设备状态
        const isConnected = Math.random() > 0.2;
        
        // 获取或创建状态元素
        let statusEl = card.querySelector('.connection-status');
        
        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.className = 'connection-status';
            card.appendChild(statusEl);
        }
        
        // 设置连接状态
        if (isConnected) {
            statusEl.innerHTML = '<i class="fas fa-circle"></i> 在线';
            statusEl.style.color = '#2ecc71';
        } else {
            statusEl.innerHTML = '<i class="fas fa-circle"></i> 离线';
            statusEl.style.color = '#e74c3c';
        }
    });
}