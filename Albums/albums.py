class Album:

    def __init__(self, data: dict[str, any]):
        self.id             =   data.get("id") or ""
        self.title          =   data.get("title") or ""
        self.artist         =   data.get("artist") if isinstance(data.get("artist"), list) else [data.get("artist")] if data.get("artist") else []
        self.date   =   data.get("date") or ""
        self.genres         =   data.get("genres") or []
        self.tags           =   data.get("tags") or []
        self.source         =   data.get("source") or ""

    def __repr__(self):
        return f"Album(title={self.title!r}, artist={self.artist!r})"
