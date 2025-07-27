# Recommendation System

## Introduction
This project implements two common types of recommendation algorithms

1. Collaborative Filtering
It recommends items to a user based on the preferences and behavorious of <i>similar users</i>. It operates under the assumption that if two users liked similar items in the past, they will likely enjoy similar items in the future. It does not require item metadata and purely relies on the user-item interaction matrix (e.g., ratings, likes).
### Pros
- Learns user preferences without needing domain knowledge
- Can uncover unexpected or novel items through user behaviour patterns
### Cons
- Suffers from the "cold start" problem (new users or items)
- Needs a sufficient amount of user interaction data to be effective

2. Content-Based Filtering
It recommends items to a user based on the features of the items themselves and the user's historical preferences. It builts a user profile by analyzing what the user liked in the past (e.g. genre, tags, actors in the movies) and recommends items with similar attributes.
### Pros
- Works well for users with unique tastes or sparse data
- Can recommend new or unpopular items if their features match the user's profile
### Cons
- Limited to recommending items similar to what the user has already liked (serendipity is lower)
- Requires rich metadata for items
