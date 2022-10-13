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

var jcp;

img = document.getElementById("imgCrop")
if (img != null) {
    jcp = Jcrop.attach("imgCrop")
}

btnTeste.onclick = evt => {
    setCanvasAndDownloadImage()
}

function getJcropCoords() {
    if (!jcp.active)
        return

    return jcp.active.pos;
}

function setCanvasAndDownloadImage() {
    var imageObj = document.getElementById("imgCrop");
    var canvas = document.getElementById("canvas");

    let coords = getJcropCoords();

    if (coords != null) {
        canvas.width = coords.w;
        canvas.height = coords.h;

        var context = canvas.getContext("2d");
        context.drawImage(imageObj, coords.x, coords.y, coords.w, coords.h, 0, 0, canvas.width, canvas.height);

        
        download(getImgValue(), "imagemCortada.png")
    }
}

function getImgValue() {
    var canvas = document.getElementById("canvas");
    var imgValue = canvas.toDataURL('image/png');

    return imgValue;
}

let download = (content, filename) => {
    let link = document.createElement('a');

    link.setAttribute('href', 'data:application/octet-stream;base64' + content);
    link.setAttribute('download', filename);

    let event = new MouseEvent('click');
    link.dispatchEvent(event);
};