async function uploadImage() {
  const input = document.getElementById('imageInput');
  if (!input.files.length) {
    alert('Please select an image.');
    return;
  }

  const formData = new FormData();
  formData.append('image', input.files[0]);

  const res = await fetch('/predict', {
    method: 'POST',
    body: formData
  });

  const data = await res.json();
  if (data.error) {
    document.getElementById('result').textContent = data.error;
  } else {
    document.getElementById('result').innerHTML = `
      Prediction: <strong>${data.prediction}</strong><br>
      Confidence: <strong>${data.confidence}</strong><br>
      <img src="/uploads/${data.image}" style="max-width: 300px; margin-top: 10px;">
    `;
  }
}

function captureImage() {
  const video = document.getElementById('camera');
  const canvas = document.getElementById('canvas');
  video.style.display = 'block';

  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
    });

  setTimeout(() => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append('image', blob, 'capture.png');

      const res = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (data.prediction) {
        document.getElementById('result').innerHTML = `
          Prediction: <strong>${data.prediction}</strong><br>
          Confidence: <strong>${data.confidence}</strong><br>
          <img src="/uploads/${data.image}" style="max-width: 300px; margin-top: 10px;">
        `;
      }
    });
  }, 3000);
}
