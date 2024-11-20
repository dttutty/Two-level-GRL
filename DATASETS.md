# Dataset provenance

The repository's curated inputs live in `datasets/`. Large source downloads are
kept locally in `leizhao_datasets/` and are intentionally ignored by Git.

## Foursquare / WWW 2019

- Source: <https://sites.google.com/site/yangdingqi/home/foursquare-dataset>
- Paper: Dingqi Yang, Bingqing Qu, Jie Yang, and Philippe Cudre-Mauroux,
  "Revisiting User Mobility and Social Relationships in LBSNs: A Hypergraph
  Embedding Approach," WWW 2019.
- The published dataset contains 22,809,624 check-ins from 114,324 users at
  3,820,891 venues, plus old and new friendship snapshots.
- `dataset_WWW_Checkins_anonymized.txt` contains anonymized user ID, venue ID,
  UTC timestamp, and timezone offset.
- `dataset_WWW_friendship_old.txt` and `dataset_WWW_friendship_new.txt` contain
  friendship pairs.
- The raw release contains `raw_Checkins_anonymized.txt` and `raw_POIs.txt`;
  POI fields include venue ID, latitude, longitude, category, and country code.

## Flickr

- Dataset notes: <https://renchi.ac.cn/datasets/>
- Upstream data used by CAN: <https://github.com/mengzaiqiao/CAN/tree/master/data>
- PyTorch Geometric reference: <https://pytorch-geometric.readthedocs.io/en/2.3.0/generated/torch_geometric.datasets.AttributedGraphDataset.html>
