/*
 * Flot plugin to export plot as svg.
 * 
 * Created by James Rising, Jan 2014
 */

(function($) {
    function init(plot) {
        // At this point, canvases haven't been setup, so wait until drawBackground
        $.plot.exportSVG = CanvasToSVG.exportSVG;
        plot.hooks.drawBackground.push(function(plot, context) { CanvasToSVG.setupCanvas(context); });
    }

    var options = {
        svg: {
        }
    };

    $.plot.plugins.push({
        init: init,
        options: options,
        name: 'svg',
        version: '1.0'
    });
})(jQuery);