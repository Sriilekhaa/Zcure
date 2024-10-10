let isDetectionEnabled = false;

chrome.storage.local.get(['detectionEnabled'], function (result) {
  isDetectionEnabled = result.detectionEnabled;
  if (isDetectionEnabled) {
    startFaceRecognition();
  }
});

chrome.storage.onChanged.addListener(function (changes) {
  if (changes.detectionEnabled) {
    isDetectionEnabled = changes.detectionEnabled.newValue;
    if (isDetectionEnabled) {
      startFaceRecognition();
    } else {
      stopFaceRecognition();
    }
  }
});

function startFaceRecognition() {
  navigator.mediaDevices.getUserMedia({ video: true }).then(function (stream) {
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();

    video.onloadeddata = () => {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      setInterval(() => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Use OpenCV or face detection algorithm here
        console.log('Running face detection...');
        // Example: Lock keyboard if face not detected
      }, 1000);
    };
  }).catch(error => {
    console.error('Error accessing camera:', error);
  });
}

function stopFaceRecognition() {
  const tracks = video.srcObject.getTracks();
  tracks.forEach(track => track.stop());
  video.srcObject = null;
}
