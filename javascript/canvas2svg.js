CanvasToSVG = {
    allOperations: [],

    // modifies canvas to capture all draw commands
    setupCanvas: function(context) {
        if (!context.isCaptured) {
            CanvasToSVG.captureMethod(context, 'save');
            CanvasToSVG.captureMethod(context, 'restore');
            CanvasToSVG.captureMethod(context, 'translate');
            CanvasToSVG.captureMethod(context, 'rotate');
            CanvasToSVG.captureMethod(context, 'beginPath');
            CanvasToSVG.captureMethod(context, 'moveTo');
            CanvasToSVG.captureMethod(context, 'lineTo');
            CanvasToSVG.captureMethod(context, 'stroke');
            CanvasToSVG.captureMethod(context, 'fill');
            CanvasToSVG.captureMethod(context, 'strokeRect');
            CanvasToSVG.captureMethod(context, 'fillRect');
            CanvasToSVG.captureMethod(context, 'clearRect');
            CanvasToSVG.captureMethod(context, 'arc');
            context.isCaptured = true;
        }
    },

    captureMethod: function(context, method) {
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
            CanvasToSVG.allOperations.push([method, this, arguments, state]);
            savedMethod.apply(this, arguments);
        };
    },

    exportSVG: function(width, height) {
        // TODO: we will need to open up a new page later, or dialog with fabricjs loaded
        var fabvas = new fabric.Canvas('c');
        fabvas.setDimensions({width: width, height: height});
        
        var orientationStack = [[1, 0, 0, 1, 0, 0]]; // a, b, c, d, e, f
        var currentPath = [];
        for (var ii = 0; ii < CanvasToSVG.allOperations.length; ii++) {
            var method = CanvasToSVG.allOperations[ii][0];
            var args = CanvasToSVG.allOperations[ii][2];
            var state = CanvasToSVG.allOperations[ii][3];

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
                orientationStack[0] = CanvasToSVG.transformRotate(matrix, args[0]);
                break;
            case 'beginPath':
                currentPath = [];
                break;
            case 'moveTo':
                currentPath.push({points: [{x: CanvasToSVG.transformPointX(orientationStack[0], args[0], args[1]), y: CanvasToSVG.transformPointY(orientationStack[0], args[0], args[1])}]});
                break;
            case 'lineTo':
                currentPath[currentPath.length-1].points.push({x: CanvasToSVG.transformPointX(orientationStack[0], args[0], args[1]), y: CanvasToSVG.transformPointY(orientationStack[0], args[0], args[1])});
                break;
            case 'stroke':
                while (currentPath.length > 0) {
                    var polyline = currentPath.shift();
                    minx = miny = Infinity;
                    for (var jj = 0; jj < polyline.points.length; jj++) {
                        minx = Math.min(polyline.points[jj].x, minx);
                        miny = Math.min(polyline.points[jj].y, miny);
                    }
                    fabvas.add(new fabric.Polyline(polyline.points, {
                        stroke: state.strokeStyle,
                        fill: 'none',
                        left: minx,
                        top: miny
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
                    left: CanvasToSVG.transformPointX(orientationStack[0], args[0], args[1]),
                    top: CanvasToSVG.transformPointY(orientationStack[0], args[0], args[1]),
                    width: args[2],
                    height: args[3]
                }));
                break;
            case 'fillRect':
                fabvas.add(new fabric.Rect({
                    fill: state.fillStyle,
                    left: CanvasToSVG.transformPointX(orientationStack[0], args[0], args[1]),
                    top: CanvasToSVG.transformPointY(orientationStack[0], args[0], args[1]),
                    width: args[2],
                    height: args[3]
                }));
                break;
            case 'clearRect':
                fabvas.add(new fabric.Rect({
                    fill: "#fff",
                    left: CanvasToSVG.transformPointX(orientationStack[0], args[0], args[1]),
                    top: CanvasToSVG.transformPointY(orientationStack[0], args[0], args[1]),
                    width: args[2],
                    height: args[3]
                }));
                break;
            case 'arc':
                // TODO
            }
        }

        return fabvas.toSVG();
    },

    transformRotate: function(matrix, angle) {
        new0 = Math.cos(angle) * matrix[0] + Math.sin(angle) * matrix[1];
        new1 = -Math.sin(angle) * matrix[0] + Math.cos(angle) * matrix[1];
        new2 = Math.cos(angle) * matrix[2] + Math.sin(angle) * matrix[3];
        new3 = -Math.sin(angle) * matrix[2] + Math.cos(angle) * matrix[3];
        new4 = Math.cos(angle) * matrix[4] + Math.sin(angle) * matrix[5];
        new5 = -Math.sin(angle) * matrix[4] + Math.cos(angle) * matrix[5];
        return [new0, new1, new2, new3, new4, new5];
    },

    transformPointX: function(matrix, x, y) {
        return matrix[0] * x + matrix[2] * y + matrix[4];
    },

    transformPointY: function(matrix, x, y) {
        return matrix[1] * x + matrix[3] * y + matrix[5];
    }
};
