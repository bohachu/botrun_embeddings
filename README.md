# python3 -m pip install botrun-embeddings

```
from botrun_embeddings import create_embeddings, find_similar_files
if __name__ == '__main__':
    create_embeddings('users/cbh_cameo_tw/data')
    print(find_similar_files("主管的性別統計資料", 'users/cbh_cameo_tw/data', top_k=5))
```