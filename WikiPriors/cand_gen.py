def get_cands_by_words(candidates, query_mention, w2t):
    query_words = query_mention.split("_")
    if len(query_words) == 1:
        return

    for word in query_words:
        try:
            candidates += w2t[word]
        except KeyError:
            #print(f'No matching candidate for query-word {word} in w2t')
            pass

def get_cands_by_surface(candidates, query_mention, p2t):
    try:
        candidates += p2t[query_mention]
    except KeyError:
        #print(f'No matching candidate for query {query_mention} using full mention string')
        pass
