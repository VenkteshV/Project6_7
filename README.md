# Project6_7
1) Grouping content into standardized taxonomy: We’ll take two different data sources (say, khan academy and learncbse) and chose a particular content from these two data sources (say, one of them being questions, or both being questions) and tag them down to an standardised taxonomy.

2) Similarity level computation for each taxonomy for the content under it from different sources: We’ll look into which content (say like of khan academy or learncbse) has been tagged to which taxonomy. In the standard taxonomy we’ll take each hierarchical path and see which content from different sources has been tagged to the same taxonomy. Now, within each taxonomy we’ll compare content from each source with other sources and compute similarity measures (like Jaccard simliarity, semantic similiarity). So basically, we’re comparing content from 2 different sources based on the taxonomy they’re grouped under rather than comparing content from one source from all the content from other sources. This way we’re reducing the search space.

3) Generating similar labels for each hierarchical label (using the synonym substitution model). - will be done by Venky sir

4) Productionising the existing model: (Will discuss with ExtraMarks tomorrow): Train the existing TagRec ++ model, fine tune it and then productionize/ make it deployable.
Backend: Venky
Front end: One of us
