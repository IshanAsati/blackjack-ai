"""Small local web server for playing blackjack with DQN advice."""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import os
import secrets
import socket
import threading
from urllib.parse import parse_qs, urlparse
import uuid

import torch

from .agent import load_policy
from .game import Game
from .table import MultiTable

STATIC = Path(__file__).with_name("static")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}


class GameService:
    def __init__(self, checkpoint: Path):
        self.agent = load_policy(checkpoint)
        self.games: dict[str, Game] = {}
        self.lock = threading.Lock()

    @torch.no_grad()
    def payload(self, game_id: str, game: Game) -> dict:
        result = game.public_state()
        result["id"] = game_id
        if game.status == "playing":
            values = self.agent.values(game.observation).tolist()
            result["advice"] = {
                "action": "hit" if values[1] > values[0] else "stand",
                "q_stand": values[0],
                "q_hit": values[1],
                "margin": abs(values[1] - values[0]),
            }
        else:
            result["advice"] = None
        return result

    def new(self) -> dict:
        with self.lock:
            game_id = uuid.uuid4().hex
            game = Game()
            self.games[game_id] = game
            # Bound memory in long-running local sessions.
            if len(self.games) > 1_000:
                self.games.pop(next(iter(self.games)))
            return self.payload(game_id, game)

    def act(self, game_id: str, action: str) -> dict:
        with self.lock:
            game = self.games.get(game_id)
            if game is None:
                raise KeyError("hand not found; deal a new hand")
            if action == "hit":
                game.hit()
            elif action == "stand":
                game.stand()
            else:
                raise ValueError("action must be hit or stand")
            return self.payload(game_id, game)

    @torch.no_grad()
    def advise(self, player: list, dealer: int) -> dict:
        """Hit/stand advice for hand-entered cards (advisor page).

        player: 2+ card values 1 (ace) to 10. dealer: one value 1-10.
        Same unlimited-deck, even-money rules the DQN trained on; pairs are
        advised as hit/stand only (no splitting in this model).
        """
        from .game import hand_value
        if isinstance(player, bool) or not isinstance(player, list) or len(player) < 2:
            raise ValueError("player must be a list of at least 2 cards")
        for card in player:
            if isinstance(card, bool) or not isinstance(card, int) or not 1 <= card <= 10:
                raise ValueError("cards must be whole numbers from 1 (ace) to 10")
        if isinstance(dealer, bool) or not isinstance(dealer, int) or not 1 <= dealer <= 10:
            raise ValueError("dealer must be a whole number from 1 (ace) to 10")
        total, usable_ace = hand_value(list(player))
        if total > 21:
            return {"player": list(player), "dealer": dealer, "player_total": total,
                    "usable_ace": bool(usable_ace), "status": "bust", "advice": None,
                    "note": "Over 21 — the hand is bust, no decision left."}
        values = self.agent.values((total, dealer, int(usable_ace))).tolist()
        # Hitting 21 always busts, so stand dominates whatever the net says.
        action = "stand" if total == 21 else ("hit" if values[1] > values[0] else "stand")
        return {"player": list(player), "dealer": dealer, "player_total": total,
                "usable_ace": bool(usable_ace),
                "status": "blackjack" if total == 21 else "playing",
                "advice": {"action": action, "q_stand": values[0], "q_hit": values[1],
                           "margin": abs(values[1] - values[0])},
                "note": ("21 — stand." if total == 21 else
                         "Pair entered: advised as hit/stand only; this model never splits."
                         if len(player) == 2 and player[0] == player[1] else None)}


class MultiService:
    """Shared table: several seats vs one dealer, DQN advice per seat."""

    def __init__(self, checkpoint: Path):
        self.agent = load_policy(checkpoint)
        self.tables: dict[str, MultiTable] = {}
        self.lock = threading.Lock()

    @torch.no_grad()
    def advice_for(self, table: MultiTable, seat: int) -> dict | None:
        if table.stage != "playing" or table.seat_status[seat] != "playing":
            return None
        values = self.agent.values(table.observation(seat)).tolist()
        return {
            "action": "hit" if values[1] > values[0] else "stand",
            "q_stand": values[0],
            "q_hit": values[1],
            "margin": abs(values[1] - values[0]),
        }

    @torch.no_grad()
    def payload(self, table_id: str, table: MultiTable) -> dict:
        state = table.public_state()
        state["id"] = table_id
        advice = {}
        for seat in range(table.num_seats):
            item = self.advice_for(table, seat)
            if item is not None:
                advice[str(seat)] = item
        state["advice"] = advice
        current = table.current_seat
        state["current_advice"] = advice.get(str(current)) if current is not None else None
        return state

    def new(self, seats: int = 2) -> dict:
        if isinstance(seats, bool) or not isinstance(seats, int) or not 1 <= seats <= 5:
            raise ValueError("seats must be an integer from 1 to 5")
        with self.lock:
            table_id = uuid.uuid4().hex
            table = MultiTable(num_seats=seats)
            self.tables[table_id] = table
            if len(self.tables) > 200:
                self.tables.pop(next(iter(self.tables)))
            return self.payload(table_id, table)

    def act(self, table_id: str, seat: int, action: str) -> dict:
        with self.lock:
            table = self.tables.get(table_id)
            if table is None:
                raise KeyError("table not found; deal a new hand")
            if isinstance(seat, bool) or not isinstance(seat, int):
                raise ValueError("seat must be an integer")
            if action == "hit":
                table.hit(seat)
            elif action == "stand":
                table.stand(seat)
            elif action == "ai":
                advice = self.advice_for(table, seat)
                if advice is None:
                    raise ValueError(f"seat {seat} has no move to play")
                if advice["action"] == "hit":
                    table.hit(seat)
                else:
                    table.stand(seat)
            else:
                raise ValueError("action must be hit, stand, or ai")
            return self.payload(table_id, table)


def handler_for(service: GameService, edge=None, multi: MultiService | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/status":
                self._json({"pid": os.getpid(), "edge_available": edge is not None,
                            "mode": "finite-shoe" if edge else "classic", "evidence": edge.bettor.evidence if edge else None,
                            "multiplayer": multi is not None})
                return
            if route == "/api/training":
                from .edge_live import training_history
                self._json(training_history())
                return
            if route == "/api/edge/session":
                try:
                    if edge is None:
                        raise ValueError("Betting model unavailable")
                    identifier = parse_qs(urlparse(self.path).query).get("id", [""])[0]
                    session = edge.get(identifier)
                    with session.lock:
                        self._json({"id": identifier, **session.payload()})
                except (KeyError, ValueError) as error:
                    self._json({"error": str(error).strip("'")}, HTTPStatus.NOT_FOUND)
                return
            filename = ("edge.html" if edge else "index.html") if route == "/" else route.removeprefix("/")
            if route == "/classic":
                filename = "index.html"
            if route == "/multi":
                filename = "multi.html"
            if route == "/advise":
                filename = "advise.html"
            if filename not in {"index.html", "styles.css", "app.js", "edge.html", "edge.css", "edge.js",
                                "multi.html", "multi.js", "multi.css",
                                "advise.html", "advise.js", "advise.css"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = STATIC / filename
            content = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", CONTENT_TYPES[path.suffix])
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4_096:
                    raise ValueError("request is too large")
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("request must be a JSON object")
                route = urlparse(self.path).path
                if route.startswith("/api/edge/"):
                    if edge is None:
                        raise ValueError("Betting model unavailable")
                    if route == "/api/edge/new":
                        response = edge.new(data.get("seed"))
                    elif route == "/api/edge/advance":
                        response = edge.advance(str(data.get("id", "")), data.get("rounds", 1), data.get("until_bet") is True)
                    else:
                        raise ValueError("Unknown edge action")
                elif route == "/api/sim/run":
                    if edge is None:
                        raise ValueError("Betting model unavailable")
                    seed = data.get("seed")
                    response = edge.bankroll_sim(data.get("bankroll_dollars", 10000),
                                                 data.get("unit_dollars", 25),
                                                 data.get("max_shoes", 1000),
                                                 seed if seed is not None else secrets.randbits(32))
                elif route == "/api/game/new":
                    response = service.new()
                elif route in {"/api/game/hit", "/api/game/stand"}:
                    response = service.act(str(data.get("id", "")), route.rsplit("/", 1)[-1])
                elif route == "/api/multi/new":
                    if multi is None:
                        raise ValueError("Multiplayer table unavailable")
                    response = multi.new(data.get("seats", 2))
                elif route in {"/api/multi/hit", "/api/multi/stand", "/api/multi/ai"}:
                    if multi is None:
                        raise ValueError("Multiplayer table unavailable")
                    response = multi.act(str(data.get("id", "")), data.get("seat", 0),
                                         route.rsplit("/", 1)[-1])
                elif route == "/api/advise":
                    response = service.advise(data.get("player", []), data.get("dealer", 0))
                else:
                    self._json({"error": "route not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._json(response)
            except (ValueError, KeyError) as error:
                self._json({"error": str(error).strip("'")}, HTTPStatus.BAD_REQUEST)

        def _json(self, data, status=HTTPStatus.OK):
            content = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    return Handler


class LocalTableServer(ThreadingHTTPServer):
    # Windows permits duplicate HTTPServer binds with SO_REUSEADDR. Claim the
    # listener exclusively so a restart cannot silently serve an older model.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(checkpoint: Path, host: str = "127.0.0.1", port: int = 8000, bettor: Path | None = None):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train a model first.")
    edge = None
    if bettor is None:
        candidate = Path("runs/edge/bet_model.pt")
        bettor = candidate if candidate.is_file() else None
    if bettor is not None:
        from .edge_live import EdgeService
        edge = EdgeService(bettor)
    server = LocalTableServer((host, port), handler_for(GameService(checkpoint), edge, MultiService(checkpoint)))
    print(f"Blackjack table: http://{host}:{port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
