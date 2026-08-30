import os, ast
p = os.path.expanduser("~/stream_server.py")
s = open(p).read()

STATE = """        self.buf = np.zeros(0, np.float32)
        self.voiced_ms = 0.0
        self.since_check = 0.0
        self.since_voice = 0.0
        self.speaking = False
        self.reply = None
        self.history = []
        self.t_speech_end = 0.0
"""

# pull the stranded state out of save_wav
stray = """        sf.write(os.path.join(self.dir, name), np.asarray(pcm, np.float32), sr)
""" + STATE
assert stray in s, "stray block not found"
s = s.replace(stray, """        sf.write(os.path.join(self.dir, name), np.asarray(pcm, np.float32), sr)
""")

# and put it back where it belongs, at the top of __init__
anchor = """        self.ws, self.http, self.loop = ws, sess, loop
        self.turn_no = 0
"""
assert anchor in s, "init anchor not found"
s = s.replace(anchor, """        self.ws, self.http, self.loop = ws, sess, loop
""" + STATE + """        self.turn_no = 0
""")
open(p, "w").write(s)

# verify: every attribute feed() touches must be assigned in __init__
tree = ast.parse(s)
cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Session")
init = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__")
assigned = {t.attr for n in ast.walk(init) if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Attribute)}
need = {"buf","voiced_ms","since_check","since_voice","speaking","reply","history","t_speech_end","turn_no","dir"}
missing = need - assigned
print("__init__ assigns:", sorted(assigned))
print("MISSING:", sorted(missing) if missing else "none")
assert not missing, missing
print("SESSION OK")
