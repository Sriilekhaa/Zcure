let detectionEnabled = false;

async function sendRequest(url) {
  try {
    const response = await fetch(url);
    const data = await response.json();
    console.log('Response from server:', data);
    return data;
  } catch (error) {
    console.error('Error:', error);
  }
}

chrome.storage.local.get(['detectionEnabled'], async function (result) {
  detectionEnabled = result.detectionEnabled || false;
  if (detectionEnabled) {
    await sendRequest('http://127.0.0.1:5000/start');
  } else {
    await sendRequest('http://127.0.0.1:5000/stop');
  }
});

chrome.runtime.onMessage.addListener(async function (request, sender, sendResponse) {
  if (request.action === 'toggleDetection') {
    detectionEnabled = !detectionEnabled;
    await chrome.storage.local.set({ detectionEnabled: detectionEnabled });
    if (detectionEnabled) {
      await sendRequest('http://127.0.0.1:5000/start');
    } else {
      await sendRequest('http://127.0.0.1:5000/stop');
    }
  }
});
