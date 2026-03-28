const personInput = document.getElementById('personImage');
const garmentInput = document.getElementById('garmentImage');
const personPreview = document.getElementById('personPreview');
const garmentPreview = document.getElementById('garmentPreview');
const personBox = document.getElementById('personBox');
const garmentBox = document.getElementById('garmentBox');
const generateBtn = document.getElementById('generateBtn');
const statusDiv = document.getElementById('status');
const outputDiv = document.getElementById('output');
const resultImage = document.getElementById('resultImage');
const downloadBtn = document.getElementById('downloadBtn');
const ratingSection = document.getElementById('ratingSection');
const rateBtn = document.getElementById('rateBtn');
const occasionSelect = document.getElementById('occasionSelect');
const ratingResult = document.getElementById('ratingResult');

// State
let lastResultImage = null;

// API Base URL
const API_URL = 'http://127.0.0.1:5000';

// Handle person image upload
personInput.addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  console.log('Person image selected:', file.name);
  
  const reader = new FileReader();
  reader.onload = function(event) {
    personPreview.src = event.target.result;
    personPreview.classList.add('show');
    personBox.classList.add('has-image');
    checkReady();
  };
  reader.readAsDataURL(file);
});

// Handle garment image upload
garmentInput.addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  console.log('Garment image selected:', file.name);
  
  const reader = new FileReader();
  reader.onload = function(event) {
    garmentPreview.src = event.target.result;
    garmentPreview.classList.add('show');
    garmentBox.classList.add('has-image');
    checkReady();
  };
  reader.readAsDataURL(file);
});

// Check if both images are uploaded
function checkReady() {
  if (personInput.files.length > 0 && garmentInput.files.length > 0) {
    generateBtn.disabled = false;
    console.log('✅ Both images ready');
  }
}

// Generate Try-On
generateBtn.addEventListener('click', async function() {
  if (!personInput.files[0] || !garmentInput.files[0]) {
    showStatus('Please upload both images', 'error');
    return;
  }
  
  const formData = new FormData();
  formData.append('person', personInput.files[0]);
  formData.append('garment', garmentInput.files[0]);
  
  try {
    generateBtn.disabled = true;
    showStatus('🎨 Generating your virtual try-on... This may take 30-60 seconds', 'loading');
    outputDiv.classList.remove('show');
    ratingSection.style.display = 'none';
    
    // Hide recommendations from any previous run
    document.getElementById('recommendationsSection').style.display = 'none';
    
    console.log('Sending request to API...');
    
    const response = await fetch(`${API_URL}/tryon`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('✅ Try-on generated successfully');
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    // Display result
    lastResultImage = data.image;
    resultImage.src = `data:image/png;base64,${data.image}`;
    outputDiv.classList.add('show');
    ratingSection.style.display = 'block';
    
    showStatus('✨ Try-on generated successfully!', 'success');
    generateBtn.disabled = false;
    
    // Scroll to result
    setTimeout(() => {
      outputDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
    
  } catch (error) {
    console.error('❌ Error:', error);
    showStatus(`Error: ${error.message}. Please try again.`, 'error');
    generateBtn.disabled = false;
  }
});

// Download Result
downloadBtn.addEventListener('click', function() {
  if (!lastResultImage) {
    showStatus('No image to download', 'error');
    return;
  }
  
  const link = document.createElement('a');
  link.href = `data:image/png;base64,${lastResultImage}`;
  link.download = `virtual-tryon-${Date.now()}.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  showStatus('📥 Image downloaded!', 'success');
});

// Rate Try-On
rateBtn.addEventListener('click', async function() {
  if (!lastResultImage) {
    showStatus('Generate a try-on first', 'error');
    return;
  }
  
  const occasion = occasionSelect.value;
  
  try {
    rateBtn.disabled = true;
    showStatus('⭐ Analyzing your try-on...', 'loading');
    
    console.log('Requesting rating for occasion:', occasion);
    
    const response = await fetch(`${API_URL}/rate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ occasion })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('✅ Rating received:', data);
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    // 1. Display rating scores
    displayRating(data);
    
    // 2. Display the 2 AI Recommendations using your existing CSS classes
    const recsSection = document.getElementById('recommendationsSection');
    const recsGrid = document.getElementById('recommendationsGrid');
    
    if (data.recommendations && data.recommendations.length > 0) {
      recsGrid.innerHTML = ''; // Clear old ones
      data.recommendations.forEach(rec => {
        recsGrid.innerHTML += `
          <div class="rec-card">
            <img src="${rec.image}" alt="${rec.name}" class="rec-image">
            <div class="rec-title">${rec.name}</div>
            <div class="rec-tags">
              <span class="rec-tag">${rec.color}</span>
              <span class="rec-tag">${rec.style}</span>
            </div>
          </div>
        `;
      });
      recsSection.style.display = 'block';
      
      // Smoothly scroll down so the reviewers see the recommendations appear
      setTimeout(() => {
        recsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 300);
    }
    
    showStatus('⭐ Rating complete!', 'success');
    rateBtn.disabled = false;
    
  } catch (error) {
    console.error('❌ Rating error:', error);
    showStatus(`Error: ${error.message}`, 'error');
    rateBtn.disabled = false;
  }
});

// Display Rating Results
function displayRating(data) {
  const { color_rating, style_rating, overall_rating, comment, detected_features } = data;
  
  ratingResult.innerHTML = `
    <div class="rating-card">
      <h4>🎯 Overall Rating</h4>
      <div class="rating-score overall">${overall_rating}/10</div>
      <div class="rating-bar">
        <div class="rating-fill" style="width: ${overall_rating * 10}%"></div>
      </div>
    </div>
    
    <div class="rating-grid">
      <div class="rating-card">
        <h4>🎨 Color Match</h4>
        <div class="rating-score">${color_rating}/10</div>
        <div class="rating-bar">
          <div class="rating-fill" style="width: ${color_rating * 10}%"></div>
        </div>
      </div>
      
      <div class="rating-card">
        <h4>👔 Style Match</h4>
        <div class="rating-score">${style_rating}/10</div>
        <div class="rating-bar">
          <div class="rating-fill" style="width: ${style_rating * 10}%"></div>
        </div>
      </div>
    </div>
    
    <div class="comment-card">
      <div class="comment-icon">💬</div>
      <p class="comment-text">${comment}</p>
    </div>
    
    <div class="details-card">
      <h4>🔍 Detected Features</h4>
      <div class="features-grid">
        <div class="feature-item">
          <span class="feature-label">Occasion:</span>
          <span class="feature-value">${detected_features.occasion}</span>
        </div>
        <div class="feature-item">
          <span class="feature-label">Color:</span>
          <span class="feature-value">${detected_features.color}</span>
        </div>
        
        <!-- <div class="feature-item">
          <span class="feature-label">Skin Tone:</span>
          <span class="feature-value">${detected_features.skin_tone}</span>
        </div> -->
        <div class="feature-item">
          <span class="feature-label">Style:</span>
          <span class="feature-value">${detected_features.style}</span>
        </div>
      </div>
    </div>
  `;
}

//Show Status Message
function showStatus(message, type) {
  statusDiv.textContent = message;
  statusDiv.className = 'status';
  
  if (type === 'error') {
    statusDiv.classList.add('error');
  } else if (type === 'success') {
    statusDiv.classList.add('success');
  }
  
  // Auto-clear success messages
  if (type === 'success') {
    setTimeout(() => {
      statusDiv.textContent = '';
      statusDiv.className = 'status';
    }, 3000);
  }
}

// Check server status on load
window.addEventListener('load', async function() {
  try {
    const response = await fetch(`${API_URL}/`);
    const data = await response.json();
    console.log('🚀 Server status:', data);
    
    if (data.status === 'running') {
      console.log('✅ Backend connected successfully');
    }
  } catch (error) {
    console.warn('⚠️ Could not connect to backend. Make sure the Flask server is running.');
    showStatus('⚠️ Backend server not running. Please start the Flask server.', 'error');
  }
});