# 新的前端HTML模板，参考sample.html风格
# 使用方式：将cloud_server.py中的dashboard()函数内容替换为以下内容

NEW_DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>边云协同智能巡检系统</title>
    <link rel="icon" type="image/png" href="https://img.icons8.com/fluency/48/cloud.png">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* 侧边栏样式 - 参考sample.html */
        .sidebar {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            height: 90vh;
            overflow-y: auto;
        }

        .sidebar h5 {
            margin-bottom: 15px;
            border-bottom: 1px solid #eaeaea;
            padding-bottom: 10px;
            color: #0d6efd;
        }

        /* 边缘服务器卡片 */
        .edge-server-card {
            background: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin-bottom: 15px;
            overflow: hidden;
            transition: all 0.3s;
        }

        .edge-server-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .edge-server-header {
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .edge-server-header.has-alert {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .edge-server-header.offline {
            background: #6c757d;
        }

        .edge-status-badge {
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(255,255,255,0.2);
        }

        /* 端侧设备列表 */
        .device-list-container {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }

        .device-list-container.expanded {
            max-height: 2000px;
            transition: max-height 0.5s ease-in;
        }

        .device-item {
            border-bottom: 1px solid #f0f0f0;
        }

        .device-item:last-child {
            border-bottom: none;
        }

        .device-header {
            padding: 12px 15px;
            background: #fafafa;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }

        .device-header:hover {
            background: #f0f0f0;
        }

        .device-header.has-alert {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }

        .device-name {
            font-weight: 600;
            color: #333;
        }

        .device-status {
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 12px;
        }

        .device-status.online {
            background: #d4edda;
            color: #155724;
        }

        .device-status.offline {
            background: #f8d7da;
            color: #721c24;
        }

        .device-status.warning {
            background: #fff3cd;
            color: #856404;
        }

        /* 报警记录列表 */
        .alert-list-container {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: white;
        }

        .alert-list-container.expanded {
            max-height: 1000px;
            transition: max-height 0.4s ease-in;
        }

        .alert-item {
            padding: 12px 15px 12px 40px;
            border-bottom: 1px solid #f5f5f5;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 12px;
            transition: background 0.2s;
        }

        .alert-item:hover {
            background: #f8f9fa;
        }

        .alert-item:last-child {
            border-bottom: none;
        }

        .alert-thumb {
            width: 60px;
            height: 45px;
            object-fit: cover;
            border-radius: 5px;
        }

        .alert-info {
            flex: 1;
        }

        .alert-title {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 3px;
        }

        .alert-time {
            font-size: 11px;
            color: #6c757d;
        }

        .alert-badge {
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }

        .alert-badge.danger {
            background: #f8d7da;
            color: #721c24;
        }

        .alert-badge.warning {
            background: #fff3cd;
            color: #856404;
        }

        .alert-badge.normal {
            background: #d4edda;
            color: #155724;
        }

        /* 展开/收起图标 */
        .expand-icon {
            transition: transform 0.3s;
        }

        .expand-icon.rotated {
            transform: rotate(180deg);
        }

        /* 主内容区 */
        .main-container {
            background-color: #fff;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            height: 90vh;
            display: flex;
            flex-direction: column;
        }

        .main-header {
            padding: 20px;
            border-bottom: 1px solid #eaeaea;
            background-color: #f8f9fa;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .main-header h4 {
            margin: 0;
            color: #333;
        }

        .ai-status-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }

        .ai-online {
            background: rgba(40, 167, 69, 0.1);
            color: #28a745;
            border: 1px solid #28a745;
        }

        .ai-offline {
            background: rgba(220, 53, 69, 0.1);
            color: #dc3545;
            border: 1px solid #dc3545;
        }

        /* 统计卡片 */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            padding: 20px;
            border-bottom: 1px solid #eaeaea;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            text-align: center;
        }

        .stat-label {
            font-size: 13px;
            color: #6c757d;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }

        .stat-value.danger {
            color: #dc3545;
        }

        .stat-value.primary {
            color: #0d6efd;
        }

        /* 最近报警网格 */
        .alerts-grid {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }

        .alert-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
            transition: all 0.2s;
            cursor: pointer;
        }

        .alert-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }

        .alert-card-thumb {
            width: 100%;
            height: 160px;
            object-fit: cover;
            display: block;
        }

        .alert-card-body {
            padding: 15px;
        }

        .alert-card-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }

        .alert-card-badge.danger {
            background: #f8d7da;
            color: #721c24;
        }

        .alert-card-badge.warning {
            background: #fff3cd;
            color: #856404;
        }

        .alert-card-badge.normal {
            background: #d4edda;
            color: #155724;
        }

        .box-indicator {
            display: inline-block;
            margin-left: 8px;
            color: #0d6efd;
            font-size: 12px;
        }

        .alert-card-title {
            margin-top: 10px;
            font-weight: 600;
            font-size: 14px;
            color: #333;
        }

        .alert-card-time {
            margin-top: 6px;
            font-size: 12px;
            color: #6c757d;
        }

        .alert-card-source {
            margin-top: 8px;
            font-size: 11px;
            color: #0d6efd;
        }

        /* 空状态 */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #6c757d;
        }

        .empty-state i {
            font-size: 48px;
            margin-bottom: 15px;
            color: #dee2e6;
        }

        /* Modal & Canvas - 保持原有功能 */
        #modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.95);
            z-index: 1000;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        #canvas-wrap {
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(0,0,0,0.8);
            max-width: 90vw;
            max-height: 70vh;
        }

        canvas {
            display: block;
            max-width: 100%;
            max-height: 70vh;
        }

        .info-box {
            width: 100%;
            max-width: 900px;
            background: #1a1a2e;
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #333;
            max-height: 25vh;
            overflow-y: auto;
        }

        .close-btn {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            cursor: pointer;
            z-index: 1001;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }

        @media (max-width: 768px) {
            .sidebar {
                height: auto;
                margin-bottom: 15px;
                max-height: 50vh;
            }
            .main-container {
                height: auto;
                min-height: 60vh;
            }
            .stats-row {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="row">
            <!-- 侧边栏 - 边缘服务器层级结构 -->
            <div class="col-md-4 col-lg-3 mb-3">
                <div class="sidebar">
                    <h5>
                        <i class="fas fa-network-wired me-2"></i>
                        边缘-端侧设备拓扑
                    </h5>
                    <div id="edge-servers-container">
                        <div class="text-center text-muted py-4">
                            <i class="fas fa-spinner fa-spin fa-2x mb-2"></i>
                            <p>加载中...</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 主内容区 - 报警详情 -->
            <div class="col-md-8 col-lg-9">
                <div class="main-container">
                    <div class="main-header">
                        <div>
                            <h4><i class="fas fa-cloud me-2 text-primary"></i>云端智能巡检中心</h4>
                            <small class="text-muted">基于 Qwen3-VL 的多模态分析</small>
                        </div>
                        <div id="ai-status" class="ai-status-badge ai-offline">
                            <i class="fas fa-circle me-1"></i>AI平台检测中...
                        </div>
                    </div>

                    <!-- 统计面板 -->
                    <div class="stats-row">
                        <div class="stat-card">
                            <div class="stat-label"><i class="fas fa-server me-1"></i>边缘节点</div>
                            <div class="stat-value" id="stat-edges">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label"><i class="fas fa-mobile-alt me-1"></i>在线设备</div>
                            <div class="stat-value" id="stat-devices">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label"><i class="fas fa-exclamation-triangle me-1"></i>异常告警</div>
                            <div class="stat-value danger" id="stat-alerts">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label"><i class="fas fa-crosshairs me-1"></i>今日定位</div>
                            <div class="stat-value primary" id="stat-boxes">-</div>
                        </div>
                    </div>

                    <!-- 报警网格 -->
                    <div class="alerts-grid" id="alerts-grid">
                        <div class="empty-state" style="grid-column: 1 / -1;">
                            <i class="fas fa-spinner fa-spin"></i>
                            <p>正在加载数据...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal - 保持原有的框解析功能 -->
    <div id="modal" onclick="if(event.target===this)closeModal()">
        <span class="close-btn" onclick="closeModal()">&times;</span>
        <div id="canvas-wrap">
            <canvas id="mainCanvas"></canvas>
        </div>
        <div class="info-box">
            <h4 id="m-title" style="margin-top:0; color:#58a6ff;"></h4>
            <p id="m-text" style="line-height:1.6; color:#c9d1d9; white-space:pre-wrap;"></p>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 全局数据存储
        let allRecords = [];
        let allDevices = [];
        let totalBoxes = 0;

        // 组织数据结构
        function organizeData(devices, records) {
            const edgeMap = {};
            
            devices.forEach(device => {
                const edgeId = device.edge_id || 'UNKNOWN';
                if (!edgeMap[edgeId]) {
                    edgeMap[edgeId] = {
                        edge_id: edgeId,
                        devices: [],
                        hasAlert: false,
                        offlineCount: 0
                    };
                }
                edgeMap[edgeId].devices.push(device);
                if (device.status === 'offline') {
                    edgeMap[edgeId].offlineCount++;
                }
            });

            // 为每个设备关联报警记录
            Object.values(edgeMap).forEach(edge => {
                edge.devices.forEach(device => {
                    device.records = records.filter(r => r.device_id === device.device_id);
                    device.hasAlert = device.records.some(r => r.alert_level !== 'normal');
                    device.alertCount = device.records.filter(r => r.alert_level !== 'normal').length;
                });
                edge.hasAlert = edge.devices.some(d => d.hasAlert);
            });

            return Object.values(edgeMap);
        }

        // 渲染边缘服务器层级结构
        function renderEdgeServers(edges) {
            const container = document.getElementById('edge-servers-container');
            
            if (edges.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-muted py-4">
                        <i class="fas fa-inbox fa-2x mb-2"></i>
                        <p>暂无边缘节点</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = edges.map((edge) => {
                const deviceCount = edge.devices.length;
                const alertCount = edge.devices.reduce((sum, d) => sum + d.alertCount, 0);
                const hasAlert = edge.hasAlert;
                const allOffline = edge.offlineCount === deviceCount && deviceCount > 0;

                return `
                    <div class="edge-server-card">
                        <div class="edge-server-header ${hasAlert ? 'has-alert' : ''} ${allOffline ? 'offline' : ''}" 
                             onclick="toggleEdge('${edge.edge_id}')">
                            <div>
                                <i class="fas fa-server me-2"></i>
                                <strong>${edge.edge_id}</strong>
                                <span class="ms-2" style="font-size: 12px; opacity: 0.9;">
                                    (${deviceCount}个端侧)
                                </span>
                            </div>
                            <div class="d-flex align-items-center gap-2">
                                ${alertCount > 0 ? `<span class="badge bg-warning text-dark"><i class="fas fa-bell me-1"></i>${alertCount}</span>` : ''}
                                ${allOffline ? '<span class="edge-status-badge">离线</span>' : '<span class="edge-status-badge">在线</span>'}
                                <i class="fas fa-chevron-down expand-icon" id="edge-icon-${edge.edge_id}"></i>
                            </div>
                        </div>
                        <div class="device-list-container" id="edge-devices-${edge.edge_id}">
                            ${edge.devices.map((device) => `
                                <div class="device-item">
                                    <div class="device-header ${device.hasAlert ? 'has-alert' : ''}" 
                                         onclick="toggleDevice('${edge.edge_id}', '${device.device_id}', event)">
                                        <div class="d-flex align-items-center">
                                            <i class="fas fa-mobile-alt me-2 ${device.status === 'online' ? 'text-success' : 'text-danger'}"></i>
                                            <span class="device-name">${device.device_id}</span>
                                        </div>
                                        <div class="d-flex align-items-center gap-2">
                                            ${device.alertCount > 0 ? `<span class="badge bg-danger">${device.alertCount} 告警</span>` : ''}
                                            <span class="device-status ${device.status}">${device.status === 'online' ? '在线' : '离线'}</span>
                                            <i class="fas fa-chevron-down expand-icon text-muted" 
                                               id="device-icon-${edge.edge_id}-${device.device_id}"></i>
                                        </div>
                                    </div>
                                    <div class="alert-list-container" id="alerts-${edge.edge_id}-${device.device_id}">
                                        ${device.records.length === 0 ? `
                                            <div class="alert-item text-muted">
                                                <i class="fas fa-check-circle me-2 text-success"></i>
                                                暂无报警记录
                                            </div>
                                        ` : device.records.map(record => `
                                            <div class="alert-item" onclick="showDetailFromElement(this, event)" 
                                                 data-record='${JSON.stringify(record).replace(/'/g, "&#39;")}'>
                                                <img src="${record.image_url}" class="alert-thumb" alt="">
                                                <div class="alert-info">
                                                    <div class="alert-title">${record.trigger_reason}</div>
                                                    <div class="alert-time">${record.timestamp}</div>
                                                </div>
                                                <span class="alert-badge ${record.alert_level}">
                                                    ${record.alert_level === 'danger' ? '🔥 危险' : 
                                                      record.alert_level === 'warning' ? '⚠️ 异常' : '✓ 正常'}
                                                </span>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }).join('');
        }

        // 切换边缘服务器展开/收起
        function toggleEdge(edgeId) {
            const container = document.getElementById(`edge-devices-${edgeId}`);
            const icon = document.getElementById(`edge-icon-${edgeId}`);
            container.classList.toggle('expanded');
            icon.classList.toggle('rotated');
        }

        // 切换设备展开/收起
        function toggleDevice(edgeId, deviceId, event) {
            event.stopPropagation();
            const container = document.getElementById(`alerts-${edgeId}-${deviceId}`);
            const icon = document.getElementById(`device-icon-${edgeId}-${deviceId}`);
            container.classList.toggle('expanded');
            icon.classList.toggle('rotated');
        }

        // 从元素显示详情
        function showDetailFromElement(element, event) {
            event.stopPropagation();
            const recordData = element.getAttribute('data-record');
            const record = JSON.parse(recordData);
            showDetail(record);
        }

        // 渲染主区域报警卡片
        function renderAlertCards(records) {
            const grid = document.getElementById('alerts-grid');
            
            if (records.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state" style="grid-column: 1 / -1;">
                        <i class="fas fa-check-circle"></i>
                        <h5>暂无检测记录</h5>
                        <p>系统运行正常，未发现异常</p>
                    </div>
                `;
                return;
            }

            grid.innerHTML = records.map(r => {
                const levelClass = r.alert_level === 'danger' ? 'danger' : 
                                  r.alert_level === 'warning' ? 'warning' : 'normal';
                const levelText = r.alert_level === 'danger' ? '🔥 危险' : 
                                 r.alert_level === 'warning' ? '⚠️ 异常' : '✓ 正常';
                const boxText = r.boxes.length > 0 ? `<span class="box-indicator"><i class="fas fa-crosshairs me-1"></i>${r.boxes.length}个定位</span>` : '';
                
                return `
                    <div class="alert-card" onclick='showDetail(${JSON.stringify(r).replace(/"/g, '&quot;')})'>
                        <img class="alert-card-thumb" src="${r.image_url}" alt="检测图像">
                        <div class="alert-card-body">
                            <div class="d-flex justify-content-between align-items-start">
                                <span class="alert-card-badge ${levelClass}">${levelText}</span>
                                ${boxText}
                            </div>
                            <div class="alert-card-title">${r.trigger_reason}</div>
                            <div class="alert-card-time"><i class="far fa-clock me-1"></i>${r.timestamp}</div>
                            <div class="alert-card-source">
                                <i class="fas fa-server me-1"></i>${r.edge_id} / ${r.device_id}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // 显示详情模态框 - 保持原有框解析功能
        function showDetail(r) {
            const modal = document.getElementById('modal');
            const canvas = document.getElementById('mainCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                
                ctx.drawImage(img, 0, 0);
                
                // 绘制检测框
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = Math.max(img.width / 200, 3);
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#00ff00';
                
                r.boxes.forEach((box, idx) => {
                    const [y1, x1, y2, x2] = box;
                    const rx = (x1 / 1000) * img.width;
                    const ry = (y1 / 1000) * img.height;
                    const rw = ((x2 - x1) / 1000) * img.width;
                    const rh = ((y2 - y1) / 1000) * img.height;
                    
                    ctx.strokeRect(rx, ry, rw, rh);
                    
                    ctx.fillStyle = '#00ff00';
                    ctx.font = `bold ${Math.max(14, img.width/40)}px Arial`;
                    ctx.fillText(`#${idx+1}`, rx + 4, ry - 6);
                });
                
                ctx.shadowBlur = 0;

                document.getElementById('m-title').innerText = `${r.trigger_reason} [${r.alert_level.toUpperCase()}]`;
                document.getElementById('m-text').innerText = r.llm_result || '无分析结果';
                modal.style.display = 'flex';
            };
            
            img.onerror = () => {
                alert('图像加载失败');
            };
            
            img.src = r.image_url;
        }

        function closeModal() { 
            document.getElementById('modal').style.display = 'none'; 
        }

        // ESC关闭
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });

        // 更新数据
        async function update() {
            try {
                const [rRes, sRes, dRes, hRes] = await Promise.all([
                    fetch('/api/v1/records'),
                    fetch('/api/v1/statistics'),
                    fetch('/api/v1/devices'),
                    fetch('/api/v1/health')
                ]);
                
                const rData = await rRes.json();
                const sData = await sRes.json();
                const dData = await dRes.json();
                const hData = await hRes.json();

                allRecords = rData.records || [];
                allDevices = dData.devices || [];

                // 更新AI状态
                const aiStatus = document.getElementById('ai-status');
                if (hData.swift_online) {
                    aiStatus.className = 'ai-status-badge ai-online';
                    aiStatus.innerHTML = '<i class="fas fa-circle me-1"></i>AI平台在线';
                } else {
                    aiStatus.className = 'ai-status-badge ai-offline';
                    aiStatus.innerHTML = '<i class="fas fa-circle me-1"></i>AI平台离线';
                }

                // 更新统计
                document.getElementById('stat-edges').innerText = new Set(allDevices.map(d => d.edge_id)).size;
                document.getElementById('stat-devices').innerText = sData.data.online_devices;
                document.getElementById('stat-alerts').innerText = sData.data.danger_alerts;
                
                totalBoxes = allRecords.reduce((sum, r) => sum + r.boxes.length, 0);
                document.getElementById('stat-boxes').innerText = totalBoxes;

                // 组织并渲染层级结构
                const edges = organizeData(allDevices, allRecords);
                renderEdgeServers(edges);

                // 渲染主区域报警卡片
                renderAlertCards(allRecords);

            } catch(e) { 
                console.error('Update error:', e); 
            }
        }

        // 定时更新
        setInterval(update, 3000);
        update();
    </script>
</body>
</html>
'''
