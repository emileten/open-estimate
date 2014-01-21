$(function() {
    $('#test_lines').click(function() {
        var $canvas = $('<canvas width="100" height="100"></canvas>');
        $(this).after($canvas);

        var context = $canvas[0].getContext("2d");
        CanvasToSVG.setupCanvas(context);

        context.beginPath();
        context.moveTo(10, 10);
        context.lineTo(90, 90);
        //context.lineTo(10, 90);
        //context.moveTo(90, 10);
        //context.lineTo(10, 90);
        context.stroke();

        $(this).next().after(CanvasToSVG.exportSVG(100, 100));
    });
});