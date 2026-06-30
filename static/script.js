document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const resultCard = document.getElementById('resultCard');
    const predictionResult = document.getElementById('predictionResult');
    const predictionMessage = document.getElementById('predictionMessage');
    
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
            predictionResult.textContent = `${result.prediction} tons/ha`;
            predictionResult.style.color = '#48bb78';
            predictionMessage.textContent = '✅ Prediction successful!';
            
            // Add celebratory effect
            resultCard.style.borderColor = '#48bb78';
        } else {
            predictionResult.textContent = '❌ Error';
            predictionResult.style.color = '#fc8181';
            predictionMessage.textContent = `⚠️ ${result.error}`;
            resultCard.style.borderColor = '#fc8181';
        }
    } catch (error) {
        predictionResult.textContent = '❌ Error';
        predictionResult.style.color = '#fc8181';
        predictionMessage.textContent = '⚠️ An error occurred while connecting to the server';
        resultCard.style.borderColor = '#fc8181';
    }
});

// Add event listeners to number inputs for real-time updates
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
    });
});