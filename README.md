You need to download some python modules in requirement and some of them are not fitting your env, so i recommand that you use pipenv to manage env.

Optional, run
```
pipenv shell
```

Then execute program by
```
python ./main
```

And the tcp client content is in reconition.py with functions init_TCP_conn() and recognize_cam(detector, sess, db_path)


If you want to change person to unknown, remove the pictures in /Figure and the old database da file, then execute
```
python .\initialization.py
```
to create new database
