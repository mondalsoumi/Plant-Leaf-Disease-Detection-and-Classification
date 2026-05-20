
document.getElementById('fileInput').addEventListener('change', function(event) {
    const image = document.getElementById('previewImage');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.classList.remove('d-none');
});

function openCamera() {
    alert("Camera feature coming soon!");
}
