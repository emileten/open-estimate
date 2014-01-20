/*
 * Flot plugin to export plot as svg.
 * 
 * Created by James Rising, Jan 2014
 */

(function($) {
    // TODO: make this local, after debugged
    allOperations = [];

    function init(plot) {
        // At this point, canvases haven't been setup, so wait until drawBackground
        $.plot.exportSVG = exportSVG;
        plot.hooks.drawBackground.push(setupCanvas);
    }

    var options = {
        svg: {
        }
    };

    // modifies canvas to capture all draw commands
    function setupCanvas(plot, context) {
        if (!context.isCaptured) {
            captureMethod(context, 'save');
            captureMethod(context, 'restore');
            captureMethod(context, 'translate');
            captureMethod(context, 'rotate');
            captureMethod(context, 'beginPath');
            captureMethod(context, 'moveTo');
            captureMethod(context, 'lineTo');
            captureMethod(context, 'stroke');
            captureMethod(context, 'fill');
            captureMethod(context, 'strokeRect');
            captureMethod(context, 'fillRect');
            captureMethod(context, 'clearRect');
            captureMethod(context, 'arc');
            context.isCaptured = true;
        }
    }

    function captureMethod(context, method) {
        var savedMethod = context[method];
        context[method] = function() {
            // copy the current state of the context
            state = {
                fillStyle: context.fillStyle,
                strokeStyle: context.strokeStyle,
                lineWidth: context.lineWidth,
                lineJoin: context.lineJoin,
                globalAlpha: context.globalAlpha,
                lineCap: context.lineCap,
                miterLimit: context.miterLimit,
                shadowOffsetX: context.shadowOffsetX,
                shadowOffsetY: context.shadowOffsetY,
                shadowBlur: context.shadowBlur,
                shadowColor: context.shadowColor,
                globalCompositeOperation: context.globalCompositeOperation,
                font: context.font,
                textAlign: context.textAlign,
                textBaseline: context.textBaseline
            };
            allOperations.push([method, this, arguments, state]);
            savedMethod.apply(this, arguments);
        };
    }

    function exportSVG() {
        // TODO: we will need to open up a new page later, or dialog with fabricjs loaded
        var fabvas = new fabric.Canvas('c');
        fabvas.setDimensions({width: 440, height: 320});
        
        var orientationStack = [[1, 0, 0, 1, 0, 0]]; // a, b, c, d, e, f
        var currentPath = [];
        for (var ii = 0; ii < allOperations.length; ii++) {
            var method = allOperations[ii][0];
            var args = allOperations[ii][2];
            var state = allOperations[ii][3];

            switch (method) {
            case 'save':
                orientationStack.unshift([1, 0, 0, 1, 0, 0]); //orientationStack[0]);
                break;
            case 'restore':
                orientationStack.shift();
                break;
            case 'translate':
                orientationStack[0][4] += args[0];
                orientationStack[0][5] += args[1];
                break;
            case 'rotate':
                orientationStack[0] = transformRotate(matrix, args[0]);
                break;
            case 'beginPath':
                currentPath = [];
                break;
            case 'moveTo':
                currentPath.push({points: [{x: transformPointX(orientationStack[0], args[0], args[1]), y: transformPointY(orientationStack[0], args[0], args[1])}]});
                break;
            case 'lineTo':
                currentPath[currentPath.length-1].points.push({x: transformPointX(orientationStack[0], args[0], args[1]), y: transformPointY(orientationStack[0], args[0], args[1])});
                break;
            case 'stroke':
                while (currentPath.length > 0) {
                    var polyline = currentPath.shift();
                    fabvas.add(new fabric.Polyline(polyline.points, {
                        stroke: state.strokeStyle,
                        fill: 'none'
                    }));
                }
                break;
            case 'fill':
                while (currentPath.length > 0) {
                    var polyline = currentPath.shift();
                    fabvas.add(new fabric.Polyline(polyline.points, {
                        fill: state.fillStyle
                    }));
                }
                break;
            case 'strokeRect':
                fabvas.add(new fabric.Rect({
                    stroke: state.strokeStyle,
                    fill: 'none',
                    left: transformPointX(orientationStack[0], args[0], args[1]),
                    top: transformPointY(orientationStack[0], args[0], args[1]),
                    width: args[2],
                    height: args[3]
                }));
                break;
            case 'fillRect':
                fabvas.add(new fabric.Rect({
                    fill: state.fillStyle,
                    left: transformPointX(orientationStack[0], args[0], args[1]),
                    top: transformPointY(orientationStack[0], args[0], args[1]),
                    width: args[2],
                    height: args[3]
                }));
                break;
            case 'clearRect':
                fabvas.add(new fabric.Rect({
                    fill: "#fff",
                    left: transformPointX(orientationStack[0], args[0], args[1]),
                    top: transformPointY(orientationStack[0], args[0], args[1]),
                    width: args[2],
                    height: args[3]
                }));
                break;
            case 'arc':
                // TODO
            }
        }

        return fabvas.toSVG();
    }

    function transformRotate(matrix, angle) {
        new0 = Math.cos(angle) * matrix[0] + Math.sin(angle) * matrix[1];
        new1 = -Math.sin(angle) * matrix[0] + Math.cos(angle) * matrix[1];
        new2 = Math.cos(angle) * matrix[2] + Math.sin(angle) * matrix[3];
        new3 = -Math.sin(angle) * matrix[2] + Math.cos(angle) * matrix[3];
        new4 = Math.cos(angle) * matrix[4] + Math.sin(angle) * matrix[5];
        new5 = -Math.sin(angle) * matrix[4] + Math.cos(angle) * matrix[5];
        return [new0, new1, new2, new3, new4, new5];
    }

    function transformPointX(matrix, x, y) {
        return matrix[0] * x + matrix[2] * y + matrix[4];
    }

    function transformPointY(matrix, x, y) {
        return matrix[1] * x + matrix[3] * y + matrix[5];
    }

    $.plot.plugins.push({
        init: init,
        options: options,
        name: 'svg',
        version: '1.0'
    });
})(jQuery);