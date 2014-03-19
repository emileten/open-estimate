/*
 * Flot plugin to export plot as svg.
 * 
 * Created by James Rising, Jan 2014
 */

(function($) {
    function init(plot) {
        // At this point, canvases haven't been setup, so wait until drawBackground
        $.plot.exportSVG = CanvasToSVG.exportSVG;

        plot.hooks.drawBackground.push(function(plot, context) {
            CanvasToSVG.setupCanvas(context);
            var surface = plot.getSurface(); // this is a custom added function
            var savedSurfaceRender = surface.render;
            surface.render = function() {
                savedSurfaceRender.call(surface); // cleans up layers

                var cache = surface._textCache;

		        for (var layerKey in cache) {
			        if (hasOwnProperty.call(cache, layerKey)) {
					    layerCache = cache[layerKey];

				        for (var styleKey in layerCache) {
					        if (hasOwnProperty.call(layerCache, styleKey)) {
						        var styleCache = layerCache[styleKey];

						        for (var key in styleCache) {
							        if (hasOwnProperty.call(styleCache, key)) {
								        var positions = styleCache[key].positions;

								        for (var i = 0, position; position = positions[i]; i++) {
                                            var $elt = $(position.element);
                                            var options = {
                                                left: $elt.offset().left - plot.getPlaceholder().offset().left,
                                                top: $elt.offset().top - plot.getPlaceholder().offset().top
                                            };

                                            var fontsize = $elt.css('fontSize');
                                            if (fontsize.substr(fontsize.length-2) == 'px')
                                                options.fontSize = fontsize.substr(0, fontsize.length-2);

                                            var text = new fabric.Text($elt.text(), options);
                                            CanvasToSVG.addFabricObject(text);
								        }
							        }
						        }
					        }
				        }
			        }
		        }
            };
        });
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