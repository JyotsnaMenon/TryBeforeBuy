// Get DOM elements
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

// Handle person image upload
personInput.addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(event) {
      personPreview.src = event.target.result;
      personPreview.classList.add('show');
      personBox.classList.add('has-image');
      checkBothImagesUploaded();
    };
    reader.readAsDataURL(file);
  }
});

// Handle garment image upload
garmentInput.addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(event) {
      garmentPreview.src = event.target.result;
      garmentPreview.classList.add('show');
      garmentBox.classList.add('has-image');
      checkBothImagesUploaded();
    };
    reader.readAsDataURL(file);
  }
});

// Enable generate button when both images are uploaded
function checkBothImagesUploaded() {
  if (personInput.files[0] && garmentInput.files[0]) {
    generateBtn.disabled = false;
  }
}

// Generate try-on
generateBtn.addEventListener('click', async function() {
  const personFile = personInput.files[0];
  const garmentFile = garmentInput.files[0];

  if (!personFile || !garmentFile) {
    showStatus('Please upload both images!', 'error');
    return;
  }

  // Disable button and show loading
  generateBtn.disabled = true;
  outputDiv.classList.remove('show');
  showStatus('Processing... This may take 30-60 seconds <span class="loading"></span>', '');

  // Prepare form data
  const formData = new FormData();
  formData.append('person', personFile);
  formData.append('garment', garmentFile);

  try {
    // Call backend API
    const response = await fetch('http://127.0.0.1:5000/tryon', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (response.ok && data.image) {
      // Success - display result
      resultImage.src = `data:image/png;base64,${data.image}`;
      outputDiv.classList.add('show');
      showStatus('✅ Success! Your virtual try-on is ready!', 'success');
      
      // Scroll to result
      outputDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });

    } else if (response.status === 429) {
      showStatus('❌ Rate limit exceeded. Please wait or upgrade your plan.', 'error');
    } else if (response.status === 403) {
      showStatus('❌ API authentication failed. Check your API key.', 'error');
    } else if (data.error) {
      showStatus(`❌ Error: ${data.error}`, 'error');
    } else {
      showStatus('❌ Unexpected error occurred. Please try again.', 'error');
    }

  } catch (error) {
    console.error('Error:', error);
    showStatus('❌ Connection error. Make sure your backend server is running on port 5000.', 'error');
  } finally {
    generateBtn.disabled = false;
  }
});

// Download result image
downloadBtn.addEventListener('click', function() {
  const link = document.createElement('a');
  link.href = resultImage.src;
  link.download = 'virtual-tryon-result.png';
  link.click();
});

// Helper function to show status messages
function showStatus(message, type) {
  statusDiv.innerHTML = message;
  statusDiv.className = 'status';
  if (type) {
    statusDiv.classList.add(type);
  }
}

// Click on upload box to trigger file input
personBox.addEventListener('click', function(e) {
  if (e.target !== personInput && !e.target.classList.contains('btn-upload')) {
    personInput.click();
  }
});

garmentBox.addEventListener('click', function(e) {
  if (e.target !== garmentInput && !e.target.classList.contains('btn-upload')) {
    garmentInput.click();
  }
});