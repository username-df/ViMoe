const result = document.getElementById('result');
const fileInput = document.getElementById('file-input')
const preview = document.getElementById('preview');

function displayImage(file) {
    if (preview.querySelectorAll("img")) {
        for (const img of preview.querySelectorAll("img")) {
            URL.revokeObjectURL(img.src);
            preview.textContent = "";
        }
    }

    if (file.type.startsWith("image/")) {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.alt = file.name;
        preview.appendChild(img);
    }
}

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    displayImage(file);
    const img = await createImageBitmap(file);    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    let width = img.width;
    let height = img.height;
    let aspect_ratio = width / height;
    
    let tw = 256;
    let th = 256;
    if (aspect_ratio < 1) {
        th = 256 / aspect_ratio;
    } else {
        tw = 256 * aspect_ratio;
    }

    canvas.width = tw;
    canvas.height = th;

    // Match with antialias=True in training phase
    ctx.imageSmoothingEnabled = true; 
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, tw, th)
    const base64Image = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
    
    result.textContent = 'Processing...';
    const response =  await browser.runtime.sendMessage({
        action: "DELIVER_IMAGE",
        image: base64Image
    });

    // READ MODEL OUTPUT
    result.textContent = response
});