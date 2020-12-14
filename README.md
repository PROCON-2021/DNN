# auto_instructor
DCON2021

## 開発環境

* Python3.8+Pipenv
* PyTorch

## Pipenv設定

* プロジェクトのディレクトリ内に`.venv`を作成する

`$ export PIPENV_VENV_IN_PROJECT=1`

* Lockingをしない

`$ export PIPENV_SKIP_LOCK=1`

## 運用方法

多分 ↓ のやり方がベストプラクティスです．

このリポジトリをForkしてブランチを切り開発．実装が完了したら`upstream/main`にPRを投げる -> マージされたら`origin/main`を更新（pull）

### upstreamを追加
`$ git remote add upstream https://github.com/kitamura-laboratory/auto_instructor`

### リポジトリのURLを確認
`$ git remote -V`

### upstreamの更新をoriginに持ってくる
```
$ git fetch upstream
$ git merge upstream/master
```

## コーディングルール

開発するにあたってコーディングルールを設けたいと思います．

**他にあれば追記お願いします.**

* インデント: スペース4文字


| 命名規則   | 用途 |
|------------|----------------------------|
| PascalCase | クラス名 |
| camelCase  | クラスインスタンス名 |
| snake_case | ファイル名，関数名，変数名 |
| SNAKE_CASE | 定数 |




