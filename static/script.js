document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const resultCard = document.getElementById('resultCard');
    const predictionResult = document.getElementById('predictionResult');
    const predictionMessage = document.getElementById('predictionMessage');
    const driftStatus = document.getElementById('driftStatus');
    
    // Collect data from the form
    const data = {
        region: document.getElementById('region').value,
        soil: document.getElementById('soil').value,
        crop: document.getElementById('crop').value,
        weather: document.getElementById('weather').value,
        temperature: document.getElementById('temperature').value,
        rainfall: document.getElementById('rainfall').value,
        fertilizer: document.getElementById('fertilizer').value,
        irrigation: document.getElementById('irrigation').value,
        days: document.getElementById('days').value
    };
    
    // Show loading
    predictionResult.textContent = '⏳ predicting...';
    resultCard.classList.remove('hidden');
    predictionResult.style.color = '#667eea';
    
    //  Show loading for drift
    if (driftStatus) {
        driftStatus.textContent = '⏳ Checking data drift...';
        driftStatus.style.color = '#667eea';
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Display prediction
            predictionResult.textContent = `${result.prediction} tons/ha`;
            predictionResult.style.color = '#48bb78';
            predictionMessage.textContent = '✅ Prediction successful!';
            resultCard.style.borderColor = '#48bb78';
            
            // 🆕 Display drift analysis
            if (result.drift_analysis && driftStatus) {
                const drift = result.drift_analysis;
                if (drift.drift_detected) {
                    driftStatus.textContent = ` ${drift.message}`;
                    driftStatus.style.color = '#ed8936';
                    driftStatus.style.background = '#fffaf0';
                    driftStatus.style.padding = '8px 12px';
                    driftStatus.style.borderRadius = '8px';
                    driftStatus.style.border = '1px solid #ed8936';
                } else {
                    driftStatus.textContent = ` ${drift.message}`;
                    driftStatus.style.color = '#48bb78';
                    driftStatus.style.background = '#f0fff4';
                    driftStatus.style.padding = '8px 12px';
                    driftStatus.style.borderRadius = '8px';
                    driftStatus.style.border = '1px solid #48bb78';
                }
            }
        } else {
            predictionResult.textContent = '❌ Error';
            predictionResult.style.color = '#fc8181';
            predictionMessage.textContent = `⚠️ ${result.error}`;
            resultCard.style.borderColor = '#fc8181';
            
            if (driftStatus) {
                driftStatus.textContent = '❌ Drift analysis unavailable';
                driftStatus.style.color = '#fc8181';
            }
        }
    } catch (error) {
        predictionResult.textContent = '❌ Error';
        predictionResult.style.color = '#fc8181';
        predictionMessage.textContent = '⚠️ An error occurred while connecting to the server';
        resultCard.style.borderColor = '#fc8181';
        
        if (driftStatus) {
            driftStatus.textContent = '❌ Connection error';
            driftStatus.style.color = '#fc8181';
        }
    }
});