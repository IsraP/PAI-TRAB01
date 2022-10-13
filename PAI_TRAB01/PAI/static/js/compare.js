imgInp.onchange = evt => {
    const [file] = imgInp.files
    if (file) {
        var reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function () {
            imgBase.value = reader.result;
            var img = document.createElement('img');
            img.style.maxHeight = "200px"
            img.src = reader.result
            imgaem1.innerHTML = ""
            imgaem1.appendChild(img);
        };
        reader.onerror = function (error) {
            console.log('Error: ', error);
        };
    }
}

imgInpCrop.onchange = evt => {
    const [file] = imgInpCrop.files
    if (file) {
        var reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function () {
            imgCrop.value = reader.result;
            var img = document.createElement('img');
            img.style.maxHeight = "200px"
            img.src = reader.result
            imgaem2.innerHTML = ""
            imgaem2.appendChild(img);
        };
        reader.onerror = function (error) {
            console.log('Error: ', error);
        };
    }
}
