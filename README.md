# Web-based-Network-Articles-of-Midterm-Elections-2026
A network analysis of how topic tags connect in 200 Guardian articles on the 2026 midterm elections.

**The Data**
Used the Guardian Open Platform API, which returns each article with the keyword tags the Guardian’s own editors assigned.

Key Fields the Dataset Includes:
- ‘id’: unique article identifier
- ‘webTitle’: article headline
- ‘webPublicationDate’: publication timestamp
- ‘sectionName’: desk (e.g. US news, Opinion)
- ‘tags’: editor-assigned topic labels

**Data Collection**
I wrote a Python script using requests to query the Guardian /search endpoint for the phrase “midterm elections” from January 1, 2026 to September 30, 2026 with show-tags=keyword and 50 results per page. The script pages through all results, pauses between requests, and caches the raw output to data/articles.json so the API is only queried once. The API key is stored in a .env file and excluded from Github. I used pandas for tables, networkx for the graph and centrality measures, and matplotlib for the figures.
