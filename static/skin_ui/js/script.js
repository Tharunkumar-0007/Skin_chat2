const video = document.getElementById('camera');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const activateCameraBtn = document.getElementById('activate-camera-btn');
let stream = null;
let predictedDisease = ''; // Variable to store the predicted disease name

// Activate camera on button click
activateCameraBtn.addEventListener('click', function() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(function(s) {
            stream = s; // Store the stream to stop later
            video.srcObject = stream;
            video.style.display = 'block';
            captureBtn.style.display = 'block';  // Show the capture button once camera is activated
            document.getElementById('result-message').textContent = ''; // Clear previous results
            document.getElementById('one').style.display = 'none'; // Hide Details button initially
        }).catch(function(error) {
            document.getElementById('error-message').textContent = 'Unable to access camera: ' + error.message;
        });
    } else {
        document.getElementById('error-message').textContent = 'Camera not supported by this browser.';
    }
});

// Capture image from the camera
captureBtn.addEventListener('click', function() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    // Convert the canvas image to Blob and send it to the backend
    canvas.toBlob(function(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'camera-image.jpg');

        fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('error-message').textContent = '';
            if (data.error) {
                document.getElementById('error-message').textContent = data.error;
                document.getElementById('one').style.display = 'none'; // Hide the Details button on error
            } else {
                predictedDisease = data.predicted_class; // Store the predicted disease name
                document.getElementById('result-message').textContent = 'Predicted Diseases: ' + predictedDisease;
                document.getElementById('one').style.display = 'block'; // Show the Details button
            }

            // Deactivate the camera after capturing
            video.style.display = 'none';
            captureBtn.style.display = 'none'; // Hide capture button
            if (stream) {
                let tracks = stream.getTracks();
                tracks.forEach(track => track.stop()); // Stop the camera
            }
        })
        .catch(error => {
            // Handle fetch error
            if (error.message === "Failed to fetch") {
                document.getElementById('error-message').textContent = 'Failed to connect to the server. Please check your network connection or server status.';
            } else {
                document.getElementById('error-message').textContent = 'An error occurred: ' + error.message;
            }
        });
    }, 'image/jpeg');
});

// Handle file upload
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from submitting the default way
    const formData = new FormData(this);

    fetch('http://localhost:5000/predict', { // Ensure URL is correct
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('error-message').textContent = '';
        if (data.error) {
            document.getElementById('error-message').textContent = data.error;
            document.getElementById('one').style.display = 'none'; // Hide the Details button on error
        } else {
            predictedDisease = data.predicted_class; // Store the predicted disease name
            document.getElementById('result-message').textContent = 'Predicted Class: ' + predictedDisease;
            document.getElementById('one').style.display = 'block'; // Show the Details button
        }
    })
    .catch(error => {
        document.getElementById('error-message').textContent = 'An error occurred: ' + error.message;
    });
});

// Handle "Details" link click
function change() {

    if (predictedDisease) {
        const s = "explain about " + predictedDisease;

        // Open the chat box if not already open
        const chatBox = document.querySelector('.chat-box');
        if (chatBox.style.display === 'none' || chatBox.style.display === '') {
            toggleChat();  // Open the chat box
        }

        // Instead of navigating, update the chat input with the explanation
        document.getElementById('query').value = s;

        // Auto-submit the form with the explanation
        setTimeout(() => {
            const event = new Event('submit', { bubbles: true });
            document.getElementById('query-form').dispatchEvent(event);
        }, 500);
    } else {
        document.getElementById('error-message').textContent = 'No predicted disease to show details.';
    }
}

