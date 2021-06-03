# texでのレポート作成のサンプル（ミニマム）

***

最低限度のサンプルコードです．

### 使い方

terminal で以下を入力

```sh
$ platex report.tex
```

これで `report.dvi` ファイルなどが作られる．その後，以下を入力．

```sh
$ dvipdfmx report.dvi
```

これで `report.pdf` が作られる．
