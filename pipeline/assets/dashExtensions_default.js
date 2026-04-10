window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng) {
            var a = (feature.properties && feature.properties.a !== undefined) ? feature.properties.a : 1.0;
            var r = (feature.properties && feature.properties.r !== undefined) ? feature.properties.r : 8;
            var c = (feature.properties && feature.properties.c) ? feature.properties.c : 'lime';
            return L.circleMarker(latlng, {
                radius: r,
                weight: 0,
                opacity: 0,
                fillOpacity: a,
                fillColor: c
            });
        }
    }
});