imgInp.onchange = evt => {
    const [file] = imgInp.files
    if (file) {
        var reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function () {
            teste.value = reader.result;
            formImg.submit();
        };
        reader.onerror = function (error) {
            console.log('Error: ', error);
        };
    }
}

img = document.getElementById("imgCrop")
if(img != null){
    Jcrop.attach("imgCrop")
    
}

btnTeste.onclick = evt =>{
    const r = Jcrop.Rect.create(10,10,100,100);
    const options = {};
    Jcrop.newWidget(r,options);
}