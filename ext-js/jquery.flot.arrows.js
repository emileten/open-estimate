/*
 * Flot Plugin to draw arrows at various x-values
 *
 * Created by Kevin Angstadt, June 2012
 *
 * datapoints for arrows are given by the tip of the arrow (^)
 * format is [x-coord, y-coord=0, color=black].
 */

(function ($) {

    function init(plot) { //"plugin body"
        plot.hooks.draw.push(drawArrows);

        function drawArrows(plot, canvasContext) {
            var opts = plot.getOptions().arrows;
            
            $.each(opts.data, function(key, value){
                var offset = plot.pointOffset({
                    x: value[0],
                    y: value[1] || 0
                });
                var color = value[2] || 'black';
                // This part does the little arrow drawing
                canvasContext.beginPath();
                canvasContext.moveTo(offset.left, offset.top);
                canvasContext.lineTo(offset.left - opts.arrowWidth / 2, offset.top + opts.arrowHeight);
                canvasContext.lineTo(offset.left + opts.arrowWidth / 2, offset.top + opts.arrowHeight);
                canvasContext.lineTo(offset.left, offset.top);
                canvasContext.fillStyle = color;
                canvasContext.fill();
            });
        }
    }
    
    var options = {
        arrows: {
            data: [],
            arrowHeight: 20,
            arrowWidth: 6
        }
    }
    
    $.plot.plugins.push({
        init: init,
        options: options,
        name: 'arrows',
        version: '1.0'
    });
})(jQuery);