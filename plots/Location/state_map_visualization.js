function drawMap() {
    var data = google.visualization.arrayToDataTable([
        ['State', 'No of Users'],
        ['US-NC',4],
        ['US-OH',4],
        ['US-CO',3],
        ['US-FL',3],
        ['US-TN',2],
        ['US-NY',5],
        ['US-VA',1],
        ['US-CT',1],
        ['US-NH',1],
        ['US-PA',8],
        ['US-AL',2],
        ['US-TX',10],
        ['US-WI',2],
        ['US-AZ',4],
        ['US-NJ',2],
        ['US-MO',1],
        ['US-IN',1],
        ['US-CA',10],
        ['US-IA',2],
        ['US-IL',4],
        ['US-MI',8],
        ['US-MA',3],
        ['US-ME',1],
        ['US-ID',2],
        ['US-GA',3],
        ['US-WA',3],
        ['US-KS',2],
        ['US-UT',1]
    ]);
    
    var options = {
        region: 'US',
        dataMode: 'regions',
        resolution: 'provinces'
    };
    
    var container = document.getElementById('map_canvas');
    var geomap = new google.visualization.GeoChart(container);
    geomap.draw(data, options);
};
google.load('visualization', '1', {packages:['geochart'], callback: drawMap});