let currentNetwork = null;

// Fetch data from backend API
async function fetchData(model = 'ensemble', top_k = 20) {
    try {
        const url = `http://127.0.0.1:8000/data?model=${model}&top_k=${top_k}`;
        console.log('Fetching from:', url);
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        console.log('Data received:', data);
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        const graphDiv = document.getElementById('graph');
        graphDiv.innerHTML = `
            <div class="error">
                <h3>❌ Connection Error</h3>
                <p><strong>Cannot connect to backend server.</strong></p>
                <p>Make sure the backend is running on <code>http://127.0.0.1:8000</code></p>
                <p><strong>To start backend:</strong></p>
                <ol style="text-align: left; margin-left: 20px;">
                    <li>Open terminal/command prompt</li>
                    <li>Navigate to backend folder: <code>cd backend</code></li>
                    <li>Run: <code>python app.py</code></li>
                </ol>
                <p style="color: #721c24;"><strong>Error details:</strong> ${error.message}</p>
            </div>
        `;
        throw error;
    }
}

// Switch between tabs
function openTab(evt, tabName) {
    document.querySelectorAll('.tabcontent').forEach(content => content.classList.remove('active'));
    document.querySelectorAll('.tablink').forEach(button => button.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    evt.currentTarget.classList.add('active');
}

// Render main graph visualization
function renderGraph(data) {
    const container = document.getElementById('graph');
    container.innerHTML = '';
    if (!data.graph || !data.graph.nodes || !data.graph.edges) {
        container.innerHTML = `<div class="error"><h3>No prediction graph data available</h3></div>`;
        return;
    }

    const nodes = new vis.DataSet(
        data.graph.nodes.map(n => ({
            id: n.id,
            label: n.label || n.id,
            color: n.group === 'drug' ? '#67C8FF' : '#FF9AA2',
            shape: 'dot',
            size: 24,
            font: { size: 16, color: '#222' }
        }))
    );

    const edges = new vis.DataSet(
        data.graph.edges.map(e => {
            const drugLabel = e.drug_name || e.drugname || e.from;
            const diseaseLabel = e.disease_name || e.diseasename || e.to;
            return {
                from: e.from,
                to: e.to,
                color: e.type === 'predicted' ? { color: '#FF0000' } : { color: '#999', opacity: 0.3 },
                width: e.type === 'predicted' ? 3 : 1,
                dashes: e.type === 'predicted',
                title: e.type === 'predicted'
                    ? `Predicted: ${drugLabel} → ${diseaseLabel}${e.score ? ', Score: ' + e.score.toFixed(4) : ''}`
                    : `Known: ${drugLabel} → ${diseaseLabel}`
            };
        })
    );

    const options = {
        nodes: { shape: 'dot', size: 24, borderWidth: 2 },
        edges: { smooth: { type: 'continuous' } },
        physics: { enabled: true, stabilization: { iterations: 200, updateInterval: 25 } },
        layout: { improvedLayout: true },
        interaction: { hover: true, tooltipDelay: 200, dragNodes: true, zoomView: true },
    };

    if (window.currentNetwork) window.currentNetwork.destroy();

    window.currentNetwork = new vis.Network(container, { nodes, edges }, options);
    window.currentNetwork.once('stabilizationIterationsDone', function () {
        window.currentNetwork.setOptions({ physics: false });
    });

    window.currentNetwork.on('click', params => {
        if (params.nodes.length) {
            const nodeId = params.nodes[0];
            const node = nodes.get(nodeId);
            const label = node.label || node.id || 'Unknown';
            const type = node.color === '#67C8FF' ? 'Drug' : 'Disease';
            alert(`Node: ${label} (${type})`);
        }
    });
}

// Render predictions table
function renderTable(predictions) {
    const tbody = document.querySelector('#pred-table tbody');
    tbody.innerHTML = '';
    if (!predictions || predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 40px;">No predictions available</td></tr>';
        return;
    }
    predictions.forEach((pred, idx) => {
        const score = pred.score || 0;
        const scoreClass = score > 0.8 ? 'score-high' : score > 0.6 ? 'score-medium' : 'score-low';
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td><strong>${pred.drug_name || pred.drug}</strong></td>
            <td><strong>${pred.disease_name || pred.disease}</strong></td>
            <td><span class="score-badge ${scoreClass}">${score.toFixed(4)}</span></td>
        `;
        tbody.appendChild(tr);
    });
    console.log('Table rendered:', predictions.length, 'predictions');
}

// Display statistics
function displayStats(stats) {
    const statsDiv = document.getElementById('stats');
    if (!stats) {
        statsDiv.innerHTML = '';
        return;
    }
    statsDiv.innerHTML = `
        <div class="stat-card">
            <h3>${stats.num_drugs || 0}</h3>
            <p>Drug Nodes</p>
        </div>
        <div class="stat-card">
            <h3>${stats.num_diseases || 0}</h3>
            <p>Disease Nodes</p>
        </div>
        <div class="stat-card">
            <h3>${stats.num_edges || 0}</h3>
            <p>Known Associations</p>
        </div>
        <div class="stat-card">
            <h3>${(stats.model_type || 'N/A').toUpperCase()}</h3>
            <p>Active Model</p>
        </div>
    `;
}

// -- New Table + Mini-Graph on-demand --
async function fetchDrugNeighborhood(drugId) {
    const resp = await fetch(`http://127.0.0.1:8000/drug_neighborhood?drug_id=${drugId}`);
    if (!resp.ok) throw new Error('Failed to fetch neighborhood');
    return resp.json();
}

function renderDrugTable(predictions) {
    const tbody = document.querySelector('#drug-table tbody');
    tbody.innerHTML = '';
    if (!predictions || predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:30px;">No predictions available</td></tr>';
        return;
    }
    const seen = new Set();
    let idx = 1;
    predictions.forEach(pred => {
        if (seen.has(pred.drug)) return;
        seen.add(pred.drug);
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${idx++}</td>
            <td><b>${pred.drug_name || pred.drug}</b></td>
            <td>${pred.disease_name || pred.disease}</td>
            <td><button data-drug="${pred.drug}">Show</button></td>
        `;
        tbody.appendChild(tr);
    });
    tbody.querySelectorAll('button[data-drug]').forEach(button => {
        button.onclick = async () => {
            const drugId = button.getAttribute('data-drug');
            const miniGraphContainer = document.getElementById('mini-graph');
            miniGraphContainer.innerHTML = 'Loading neighborhood...';
            try {
                const data = await fetchDrugNeighborhood(drugId);
                miniGraphContainer.innerHTML = '';
                showMiniGraph(data, miniGraphContainer);
            } catch (e) {
                miniGraphContainer.innerHTML = 'Failed to load graph.';
            }
        };
    });
}

function showMiniGraph(data, container) {
    if (!data.nodes || !data.edges) {
        container.innerHTML = 'No graph data found.';
        return;
    }
    container.innerHTML = `<div id="mini-ng" style="height: 100%;"></div>`;

    const nodes = new vis.DataSet(data.nodes.map(n => ({
        id: n.id,
        label: n.label || n.id,
        color: n.group === 'drug' ? '#67C8FF' : '#FF9AA2',
        shape: 'dot',
        size: 24,
        font: { size: 16, color: '#222' }
    })));

    const edges = new vis.DataSet(data.edges.map(e => ({
        from: e.from,
        to: e.to,
        color: e.type === 'predicted' ? { color: '#FF0000' } : { color: '#999' },
        width: e.type === 'predicted' ? 3 : 1,
        dashes: e.type === 'predicted',
        title: e.type === 'predicted'
            ? `Predicted: ${e.drug_name || e.from} → ${e.disease_name || e.to}`
            : `Known: ${e.drug_name || e.from} → ${e.disease_name || e.to}`
    })));

    const options = {
        nodes: { shape: 'dot', size: 24, borderWidth: 2 },
        edges: { smooth: { type: 'continuous' } },
        physics: {
            enabled: true,
            solver: 'barnesHut',
            barnesHut: {
                gravitationalConstant: -2000,
                centralGravity: 0.3,
                springLength: 120,
                springConstant: 0.05,
                damping: 0.09
            },
            stabilization: { iterations: 150, updateInterval: 25 }
        },
        layout: { improvedLayout: true },
        interaction: { hover: true, tooltipDelay: 200, dragNodes: true, zoomView: true }
    };

    const miniNetwork = new vis.Network(document.getElementById('mini-ng'), { nodes, edges }, options);

    // Optional: freeze physics after layout settles for smoother look
    miniNetwork.once('stabilizationIterationsDone', function () {
        miniNetwork.setOptions({ physics: false });
    });
}


// Main function to load data
async function loadData() {
    const model = document.getElementById('model-select').value;
    const topK = parseInt(document.getElementById('topk-input').value);

    if (topK < 5 || topK > 100) {
        alert('Top K must be between 5 and 100');
        return;
    }

    const graphDiv = document.getElementById('graph');
    graphDiv.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Loading predictions from ${model.toUpperCase()} model...</p>
        </div>
    `;

    try {
        const data = await fetchData(model, topK);

        const successMsg = document.createElement('div');
        successMsg.className = 'success';
        successMsg.innerHTML = `✅ Successfully loaded ${data.predictions.length} predictions from ${model.toUpperCase()} model!`;
        graphDiv.parentElement.insertBefore(successMsg, graphDiv);
        setTimeout(() => successMsg.remove(), 3000);

        renderGraph(data);
        renderTable(data.predictions);
        renderDrugTable(data.predictions);  // Renders the mini-table
        displayStats(data.stats);

        console.log('✅ Data loaded successfully');
    } catch (error) {
        console.error('❌ Error loading data:', error);
    }
}

// Test backend connection on page load
window.addEventListener('load', async () => {
    console.log('Frontend loaded. Testing backend connection...');
    try {
        const res = await fetch('http://127.0.0.1:8000/');
        if (res.ok) {
          const info = await res.json();
          console.log('✅ Backend connected:', info);
          const header = document.querySelector('header p');
          header.innerHTML += ' <span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 5px; font-size: 0.9em;">🟢 Backend Connected</span>';
        }
    } catch (error) {
        console.error('❌ Backend not connected:', error);
        const header = document.querySelector('header p');
        header.innerHTML += ' <span style="background: rgba(255,100,100,0.3); padding: 5px 10px; border-radius: 5px; font-size: 0.9em;">🔴 Backend Offline</span>';
        const graphDiv = document.getElementById('graph');
        graphDiv.innerHTML = `
            <div class="error">
                <h3>⚠️ Backend Server Not Running</h3>
                <p>The frontend cannot connect to the backend API.</p>
                <p><strong>To start the backend:</strong></p>
                <ol style="text-align: left; margin-left: 20px; margin-top: 10px;">
                    <li>Open a terminal/command prompt</li>
                    <li>Navigate to backend folder: <code>cd backend</code></li>
                    <li>Run the server: <code>python app.py</code></li>
                    <li>Wait for message: "Uvicorn running on http://127.0.0.1:8000"</li>
                    <li>Refresh this page</li>
                </ol>
            </div>
        `;
    }
});
