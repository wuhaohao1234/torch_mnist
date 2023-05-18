// (function () {
//     var canvas = document.querySelector("#canvas");
//     var context = canvas.getContext("2d");
//     canvas.width = 280;
//     canvas.height = 280;

//     var Mouse = {x: 0, y: 0};
//     var lastMouse = {x: 0, y: 0};
//     context.fillStyle = "black";
//     context.fillRect(0, 0, canvas.width, canvas.height);
//     context.color = "white";
//     context.lineWidth = 10;
//     context.lineJoin = context.lineCap = 'round';

//     debug();

//     canvas.addEventListener("mousemove", function (e) {
//         lastMouse.x = Mouse.x;
//         lastMouse.y = Mouse.y;
//         Mouse.x = e.pageX - this.offsetLeft - 15;
//         Mouse.y = e.pageY - this.offsetTop - 15;
//     }, false);

//     canvas.addEventListener("mousedown", function (e) {
//         canvas.addEventListener("mousemove", onPaint, false);
//     }, false);

//     canvas.addEventListener("mouseup", function () {
//         canvas.removeEventListener("mousemove", onPaint, false);
//     }, false);

//     var onPaint = function () {
//         context.lineWidth = context.lineWidth;
//         context.lineJoin = "round";
//         context.lineCap = "round";
//         context.strokeStyle = context.color;

//         context.beginPath();
//         context.moveTo(lastMouse.x, lastMouse.y);
//         context.lineTo(Mouse.x, Mouse.y);
//         context.closePath();
//         context.stroke();
//     };

//     function debug() {
//         $("#clearButton").on("click", function () {
//             context.clearRect(0, 0, 280, 280);
//             context.fillStyle = "black";
//             context.fillRect(0, 0, canvas.width, canvas.height);
//         });
//     }
// }());

(function () {
    var canvas = document.querySelector("#canvas");
    var context = canvas.getContext("2d");                // 返回一个对象，该对象提供了用于在画布上绘图的方法和属性。创建绘制区，用于手写数字
    canvas.width = 280;
    canvas.height = 280;
    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);  //填充矩形，设置画布位置及大小
    context.color = "white";
    context.lineWidth = 10;
    context.lineJoin ='round';
    context.lineCap = 'round';

    var Mouse = {x: 0, y: 0};                            // canvas左上角为(0, 0)
    var lastMouse = {x: 0, y: 0};

    debug();                                             // 监听所有的listener

    var drawing = false;                                           //是否绘制的标志变量

    //console.log(canvas.width)
    //console.log(canvas.height)


    function getScaleFactor() {                                    //为避免光标与字体轨迹偏移问题，故而计算canvas的比例
        var rect = canvas.getBoundingClientRect();                 //获取canvas的大小及相对位置
        return {
            scaleX: canvas.width / rect.width,                     //canvas.width=canvas.height=280  rect.width=rect.height=296
            scaleY: canvas.height / rect.height
        };
    }

    canvas.addEventListener("mousemove", function (e) {
        if (!drawing) return;
        lastMouse.x = Mouse.x;
        lastMouse.y = Mouse.y;
        var scaleFactor = getScaleFactor();
        Mouse.x = (e.clientX - canvas.getBoundingClientRect().left) * scaleFactor.scaleX;
        Mouse.y = (e.clientY - canvas.getBoundingClientRect().top) * scaleFactor.scaleY;
        onPaint();
    });

    canvas.addEventListener("mousedown", function (e) {
        drawing = true;
        var scaleFactor = getScaleFactor();
        lastMouse.x = Mouse.x;
        lastMouse.y = Mouse.y;
        Mouse.x = (e.clientX - canvas.getBoundingClientRect().left) * scaleFactor.scaleX;
        Mouse.y = (e.clientY - canvas.getBoundingClientRect().top) * scaleFactor.scaleY;

    });

    canvas.addEventListener("mouseup", function () {
        drawing = false;
    });

    var onPaint = function () {
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = context.color;

        context.beginPath();
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(Mouse.x, Mouse.y);
        context.closePath();
        context.stroke();
    };

    function debug() {                                                //debug用于设置事件监听器的函数，监听所有的listener
        $("#clearButton").on("click", function () {
            context.clearRect(0, 0, 280, 280);
            context.fillStyle = "black";
            context.fillRect(0, 0, canvas.width, canvas.height);
        });
    }
}());