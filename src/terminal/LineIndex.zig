//! Line index for page lists: which rows end a line, and where the n-th line
//! ends, without walking the list.
//!
//! This file is pure: it knows nothing of `PageList`, so its tests run fast.
//! A line end is a row whose `wrap` flag is clear.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// First line id. Ids start high so restoring pages above the first one can
/// lower the first id without underflow.
pub const LINE_ORIGIN: u64 = 1 << 48;

/// Number of u64 bitmap words that hold `rows` bits.
pub fn wordsFor(rows: usize) usize {
    return (rows + 63) / 64;
}

/// Set bit i of `out` when row i of `raw` is a line end (its `wrap_bit` is
/// clear) and return how many are set. `raw` holds one u64 per row, as
/// stored in a page; `out` needs `wordsFor(raw.len)` words, all written.
///
/// Eight rows are tested per vector and 64 per output word, which keeps a
/// page's scan around 0.2 microseconds.
pub fn scanLineEnds(raw: []const u64, wrap_bit: u6, out: []u64) u32 {
    const V = @Vector(8, u64);
    const mask: V = @splat(@as(u64, 1) << wrap_bit);
    const n = raw.len;
    var count: u32 = 0;
    var w: usize = 0;
    var i: usize = 0;
    while (i + 64 <= n) : (i += 64) {
        var word: u64 = 0;
        inline for (0..8) |g| {
            const v: V = raw[i + g * 8 ..][0..8].*;
            const ends: @Vector(8, bool) = (v & mask) == @as(V, @splat(0));
            const byte: u8 = @bitCast(ends);
            word |= @as(u64, byte) << (g * 8);
        }
        out[w] = word;
        w += 1;
        count += @popCount(word);
    }
    if (i < n) {
        var word: u64 = 0;
        for (raw[i..], 0..) |r, j| {
            if ((r >> wrap_bit) & 1 == 0) word |= @as(u64, 1) << @intCast(j);
        }
        out[w] = word;
        count += @popCount(word);
    }
    return count;
}

/// Number of rows of `raw` that are line ends, without building a bitmap.
pub fn countLineEnds(raw: []const u64, wrap_bit: u6) u32 {
    var count: u32 = 0;
    const mask = @as(u64, 1) << wrap_bit;
    for (raw) |r| count += @intFromBool(r & mask == 0);
    return count;
}

/// Number of set bits among the first `row` bits.
pub fn rankBits(bits: []const u64, row: usize) u32 {
    var total: u32 = 0;
    const full = @min(row / 64, bits.len);
    for (bits[0..full]) |w| total += @popCount(w);
    const rem: u6 = @intCast(row % 64);
    if (rem > 0 and full < bits.len) total += @popCount(bits[full] & ((@as(u64, 1) << rem) - 1));
    return total;
}

/// Position of the k-th (0-based) set bit, or null when there are fewer.
pub fn selectBits(bits: []const u64, k: u32) ?u32 {
    var left = k;
    for (bits, 0..) |word, wi| {
        const pc: u32 = @popCount(word);
        if (left < pc) {
            var w = word;
            for (0..left) |_| w &= w - 1;
            return @intCast(wi * 64 + @ctz(w));
        }
        left -= pc;
    }
    return null;
}

/// Line ends of one page: a count plus a bitmap with one bit per row.
///
/// A single heap allocation holds the header and the bitmap; it remembers its
/// allocator so a node can drop it without one at hand. Valid only for a page
/// nothing writes to: it is rebuilt, not updated.
pub const LineEnds = struct {
    alloc: Allocator,
    /// Rows without `wrap` in the page.
    count: u32,
    /// Rows summarised; a debug recount checks it against the page.
    rows: u32,
    words: u32,

    fn allocBytes(words: usize) usize {
        return @sizeOf(LineEnds) + words * @sizeOf(u64);
    }

    /// Summarise `raw` (one u64 per row; see `scanLineEnds`).
    pub fn build(alloc: Allocator, raw: []const u64, wrap_bit: u6) Allocator.Error!*LineEnds {
        const words = wordsFor(raw.len);
        const mem = try alloc.alignedAlloc(u8, .of(LineEnds), allocBytes(words));
        const self: *LineEnds = @ptrCast(mem.ptr);
        self.alloc = alloc;
        self.rows = @intCast(raw.len);
        self.words = @intCast(words);
        const out: [*]u64 = @ptrCast(@alignCast(mem.ptr + @sizeOf(LineEnds)));
        self.count = scanLineEnds(raw, wrap_bit, out[0..words]);
        return self;
    }

    /// The bitmap, `words` long.
    pub fn bits(self: *const LineEnds) []const u64 {
        const base: [*]const u8 = @ptrCast(self);
        const p: [*]const u64 = @ptrCast(@alignCast(base + @sizeOf(LineEnds)));
        return p[0..self.words];
    }

    /// Line ends in rows [0, row).
    pub fn rank(self: *const LineEnds, row: u32) u32 {
        if (row >= self.rows) return self.count;
        return rankBits(self.bits(), row);
    }

    /// Row of the k-th (0-based) line end.
    pub fn select(self: *const LineEnds, k: u32) ?u32 {
        if (k >= self.count) return null;
        return selectBits(self.bits(), k);
    }

    /// Free the summary.
    pub fn deinit(self: *LineEnds) void {
        const alloc = self.alloc;
        const mem: [*]align(@alignOf(LineEnds)) u8 = @ptrCast(self);
        alloc.free(mem[0..allocBytes(self.words)]);
    }
};

/// Entries per prefix-sum block of `NodeIndex`.
pub const block_entries = 256;

/// Flat index of a list's nodes in list order: per node its row and line-end
/// counts, with prefix sums every `block_entries` entries.
///
/// Positions are relative to the first live entry. The front moves by a head
/// offset, so dropping from it is O(1); the arrays compact once the head
/// passes half. `Ptr` is the node pointer type, kept opaque.
pub fn NodeIndex(comptime Ptr: type) type {
    return struct {
        const Self = @This();

        pub const Entry = struct { node: Ptr, rows: u16, lines: u16 };

        /// An entry's index with the rows and line ends before it.
        pub const Pos = struct { index: usize, rows_before: u64, lines_before: u64 };

        // Parallel arrays indexed from 0; entries before `head` are dropped.
        nodes: std.ArrayList(Ptr) = .empty,
        rows_: std.ArrayList(u16) = .empty,
        lines_: std.ArrayList(u16) = .empty,
        // Sums over all entries before array index `b * block_entries`.
        pre_rows: std.ArrayList(u64) = .empty,
        pre_lines: std.ArrayList(u64) = .empty,
        head: usize = 0,
        // Sums over the dropped entries.
        base_rows: u64 = 0,
        base_lines: u64 = 0,
        // Sums over all entries, dropped included.
        tot_rows: u64 = 0,
        tot_lines: u64 = 0,
        // A compaction in progress: the live entries are being copied down
        // by `move_shift` (a whole number of blocks), `moved` of them so far.
        // Readers keep using the old place until the last chunk lands.
        move_shift: usize = 0,
        moved: usize = 0,
        moving: bool = false,

        /// Free the arrays.
        pub fn deinit(self: *Self, alloc: Allocator) void {
            self.nodes.deinit(alloc);
            self.rows_.deinit(alloc);
            self.lines_.deinit(alloc);
            self.pre_rows.deinit(alloc);
            self.pre_lines.deinit(alloc);
            self.* = .{};
        }

        /// Live entries.
        pub fn len(self: *const Self) usize {
            return self.nodes.items.len - self.head;
        }

        /// Rows of all live entries.
        pub fn totalRows(self: *const Self) u64 {
            return self.tot_rows - self.base_rows;
        }

        /// Line ends of all live entries.
        pub fn totalLines(self: *const Self) u64 {
            return self.tot_lines - self.base_lines;
        }

        /// Live entry `i`.
        pub fn entry(self: *const Self, i: usize) Entry {
            const a = self.head + i;
            return .{
                .node = self.nodes.items[a],
                .rows = self.rows_.items[a],
                .lines = self.lines_.items[a],
            };
        }

        /// Add an entry at the end.
        pub fn append(self: *Self, alloc: Allocator, node: Ptr, rows: u16, lines: u16) Allocator.Error!void {
            const i = self.nodes.items.len;
            try self.nodes.ensureUnusedCapacity(alloc, 1);
            try self.rows_.ensureUnusedCapacity(alloc, 1);
            try self.lines_.ensureUnusedCapacity(alloc, 1);
            try self.pre_rows.ensureUnusedCapacity(alloc, 1);
            try self.pre_lines.ensureUnusedCapacity(alloc, 1);
            if (i % block_entries == 0) {
                self.pre_rows.appendAssumeCapacity(self.tot_rows);
                self.pre_lines.appendAssumeCapacity(self.tot_lines);
            }
            self.nodes.appendAssumeCapacity(node);
            self.rows_.appendAssumeCapacity(rows);
            self.lines_.appendAssumeCapacity(lines);
            self.tot_rows += rows;
            self.tot_lines += lines;
        }

        /// Drop every live entry from position `from` on.
        pub fn truncate(self: *Self, from: usize) void {
            if (from >= self.len()) return;
            const new_len = self.head + from;
            self.cancelMoveFrom(new_len);
            self.nodes.shrinkRetainingCapacity(new_len);
            self.rows_.shrinkRetainingCapacity(new_len);
            self.lines_.shrinkRetainingCapacity(new_len);
            const blocks = (new_len + block_entries - 1) / block_entries;
            self.pre_rows.shrinkRetainingCapacity(blocks);
            self.pre_lines.shrinkRetainingCapacity(blocks);
            self.recomputeTotals();
        }

        /// Totals over the first `items.len` entries, from the last block's sum.
        fn recomputeTotals(self: *Self) void {
            const n = self.nodes.items.len;
            if (n == 0) {
                self.tot_rows = 0;
                self.tot_lines = 0;
                return;
            }
            const b = (n - 1) / block_entries;
            var r = self.pre_rows.items[b];
            var l = self.pre_lines.items[b];
            for (b * block_entries..n) |i| {
                r += self.rows_.items[i];
                l += self.lines_.items[i];
            }
            self.tot_rows = r;
            self.tot_lines = l;
        }

        /// Rebuild the prefix sums of every block from array index `from` on,
        /// after the arrays changed there. Block sums must have capacity.
        fn recomputeFrom(self: *Self, from: usize) void {
            const n = self.nodes.items.len;
            const blocks = (n + block_entries - 1) / block_entries;
            const first_block = from / block_entries;
            // Sums of the entries before the first block to rebuild.
            var r: u64 = 0;
            var l: u64 = 0;
            if (first_block > 0) {
                const prev = first_block - 1;
                r = self.pre_rows.items[prev];
                l = self.pre_lines.items[prev];
                for (prev * block_entries..first_block * block_entries) |i| {
                    r += self.rows_.items[i];
                    l += self.lines_.items[i];
                }
            }
            self.pre_rows.items.len = @min(self.pre_rows.items.len, first_block);
            self.pre_lines.items.len = @min(self.pre_lines.items.len, first_block);
            var i = first_block * block_entries;
            while (i < n) : (i += 1) {
                if (i % block_entries == 0) {
                    self.pre_rows.appendAssumeCapacity(r);
                    self.pre_lines.appendAssumeCapacity(l);
                }
                r += self.rows_.items[i];
                l += self.lines_.items[i];
            }
            std.debug.assert(self.pre_rows.items.len == blocks);
            self.tot_rows = r;
            self.tot_lines = l;
        }

        /// Open a gap of `add` slots after removing `remove` at `s`, in one array.
        fn reshape(list: anytype, s: usize, remove: usize, add: usize) void {
            const old = list.items.len;
            const tail = old - s - remove;
            const p = list.items.ptr;
            @memmove(p[s + add ..][0..tail], p[s + remove ..][0..tail]);
            list.items.len = old - remove + add;
        }

        /// Replace `remove` entries at `pos` with `new_entries`; prefix sums
        /// are recomputed from `pos`. On error nothing changed.
        pub fn splice(self: *Self, alloc: Allocator, pos: usize, remove: usize, new_entries: []const Entry) Allocator.Error!void {
            std.debug.assert(pos + remove <= self.len());
            const s = self.head + pos;
            self.cancelMoveFrom(s);
            if (remove == new_entries.len) return self.overwrite(s, new_entries);
            const new_len = self.nodes.items.len - remove + new_entries.len;
            try self.nodes.ensureTotalCapacity(alloc, new_len);
            try self.rows_.ensureTotalCapacity(alloc, new_len);
            try self.lines_.ensureTotalCapacity(alloc, new_len);
            const blocks = (new_len + block_entries - 1) / block_entries;
            try self.pre_rows.ensureTotalCapacity(alloc, blocks);
            try self.pre_lines.ensureTotalCapacity(alloc, blocks);

            reshape(&self.nodes, s, remove, new_entries.len);
            reshape(&self.rows_, s, remove, new_entries.len);
            reshape(&self.lines_, s, remove, new_entries.len);
            for (new_entries, s..) |e, i| {
                self.nodes.items[i] = e.node;
                self.rows_.items[i] = e.rows;
                self.lines_.items[i] = e.lines;
            }
            self.recomputeFrom(s);
        }

        /// Replace as many entries as given, in place, at array index `s`: no
        /// memmove, and the block sums shift by the change instead of being
        /// rebuilt (a lazy reflow step often keeps the node count).
        fn overwrite(self: *Self, s: usize, new_entries: []const Entry) void {
            var d_rows: i64 = 0;
            var d_lines: i64 = 0;
            for (new_entries, s..) |e, i| {
                d_rows += @as(i64, e.rows) - self.rows_.items[i];
                d_lines += @as(i64, e.lines) - self.lines_.items[i];
                self.nodes.items[i] = e.node;
                self.rows_.items[i] = e.rows;
                self.lines_.items[i] = e.lines;
                // A block that starts inside the range sees the change so far.
                if ((i + 1) % block_entries == 0 and (i + 1) / block_entries < self.pre_rows.items.len) {
                    const b = (i + 1) / block_entries;
                    self.pre_rows.items[b] = @intCast(@as(i64, @intCast(self.pre_rows.items[b])) + d_rows);
                    self.pre_lines.items[b] = @intCast(@as(i64, @intCast(self.pre_lines.items[b])) + d_lines);
                }
            }
            // Blocks starting beyond it see all of it.
            const first = (s + new_entries.len) / block_entries + 1;
            for (self.pre_rows.items[@min(first, self.pre_rows.items.len)..]) |*v| v.* = @intCast(@as(i64, @intCast(v.*)) + d_rows);
            for (self.pre_lines.items[@min(first, self.pre_lines.items.len)..]) |*v| v.* = @intCast(@as(i64, @intCast(v.*)) + d_lines);
            self.tot_rows = @intCast(@as(i64, @intCast(self.tot_rows)) + d_rows);
            self.tot_lines = @intCast(@as(i64, @intCast(self.tot_lines)) + d_lines);
        }

        /// Drop the first `n` live entries in O(n). Dead entries are reclaimed
        /// in bounded chunks over the next calls, never in one long move.
        pub fn dropFront(self: *Self, n: usize) void {
            std.debug.assert(n <= self.len());
            for (self.head..self.head + n) |i| {
                self.base_rows += self.rows_.items[i];
                self.base_lines += self.lines_.items[i];
            }
            self.head += n;
            if (self.moving) return self.moveChunk();
            // Compact once more than half of the arrays is dead.
            if (self.head * 2 <= self.nodes.items.len or self.head < block_entries) return;
            const shift = self.head / block_entries * block_entries;
            const live = self.nodes.items.len - shift;
            // Small, or so dead that the memory matters more: move it at once.
            if (live <= move_chunk or self.head > 3 * self.len()) return self.compactNow();
            // The copy must land in dead entries only: no overlap with the
            // live range, or readers would see torn entries.
            if (live > self.head) return;
            self.move_shift = shift;
            self.moved = 0;
            self.moving = true;
            self.moveChunk();
        }

        /// Entries copied per call while compacting: about 100 KB, a few µs (small in tests, so they reach it).
        const move_chunk = if (@import("builtin").is_test) 64 else 8192;

        /// Forget a partial copy that an edit starting at array index `start`
        /// made stale: only the entries already copied matter.
        fn cancelMoveFrom(self: *Self, start: usize) void {
            if (self.moving and start < self.move_shift + self.moved) self.moving = false;
        }

        /// Copy the next chunk of the live entries down; the last one ends
        /// the compaction.
        fn moveChunk(self: *Self) void {
            const end = self.nodes.items.len;
            const from = self.move_shift + self.moved;
            // An append since the start may have grown the live range into
            // the destination: start over rather than corrupt it.
            if (end - self.move_shift > self.head) {
                self.moving = false;
                return;
            }
            const c = @min(move_chunk, end - from);
            @memcpy(self.nodes.items[self.moved..][0..c], self.nodes.items[from..][0..c]);
            @memcpy(self.rows_.items[self.moved..][0..c], self.rows_.items[from..][0..c]);
            @memcpy(self.lines_.items[self.moved..][0..c], self.lines_.items[from..][0..c]);
            self.moved += c;
            if (from + c >= end) {
                self.moving = false;
                self.finishCompact(self.move_shift);
            }
        }

        /// Move every live entry down at once.
        fn compactNow(self: *Self) void {
            const shift = self.head / block_entries * block_entries;
            if (shift == 0) return;
            const n = self.nodes.items.len - shift;
            @memmove(self.nodes.items[0..n], self.nodes.items[shift..][0..n]);
            @memmove(self.rows_.items[0..n], self.rows_.items[shift..][0..n]);
            @memmove(self.lines_.items[0..n], self.lines_.items[shift..][0..n]);
            self.finishCompact(shift);
        }

        /// The entries are already `shift` (a whole number of blocks) lower:
        /// switch over. Block sums only lose the removed blocks, O(blocks)
        /// instead of a rebuild over every entry.
        fn finishCompact(self: *Self, shift: usize) void {
            const k = shift / block_entries;
            const n = self.nodes.items.len - shift;
            self.nodes.items.len = n;
            self.rows_.items.len = n;
            self.lines_.items.len = n;
            const cut_rows = self.pre_rows.items[k];
            const cut_lines = self.pre_lines.items[k];
            const blocks = self.pre_rows.items.len - k;
            for (0..blocks) |b| {
                self.pre_rows.items[b] = self.pre_rows.items[b + k] - cut_rows;
                self.pre_lines.items[b] = self.pre_lines.items[b + k] - cut_lines;
            }
            self.pre_rows.items.len = blocks;
            self.pre_lines.items.len = blocks;
            self.head -= shift;
            self.base_rows -= cut_rows;
            self.base_lines -= cut_lines;
            self.tot_rows -= cut_rows;
            self.tot_lines -= cut_lines;
        }

        /// The entry holding the k-th (0-based, among live entries) line end.
        pub fn findLine(self: *const Self, k: u64) ?Pos {
            if (k >= self.totalLines()) return null;
            return self.find(self.pre_lines.items, self.lines_.items, self.base_lines, self.base_rows, k, true);
        }

        /// The entry holding row `r` (relative to the first live row).
        pub fn findRow(self: *const Self, r: u64) ?Pos {
            if (r >= self.totalRows()) return null;
            return self.find(self.pre_rows.items, self.rows_.items, self.base_rows, self.base_lines, r, false);
        }

        /// Shared search: binary search on one prefix array, then a scan of at
        /// most one block of counts. `by_lines` says which sum `key` counts.
        fn find(self: *const Self, pre: []const u64, counts: []const u16, base: u64, other_base: u64, key: u64, by_lines: bool) ?Pos {
            const target = base + key;
            var lo = self.head / block_entries;
            var hi = pre.len;
            // Last block whose sum is <= target.
            while (hi - lo > 1) {
                const mid = (lo + hi) / 2;
                if (pre[mid] <= target) lo = mid else hi = mid;
            }
            var i = lo * block_entries;
            var sum: u64 = pre[lo];
            var other: u64 = if (by_lines) self.pre_rows.items[lo] else self.pre_lines.items[lo];
            if (i < self.head) {
                i = self.head;
                sum = base;
                other = other_base;
            }
            while (i < counts.len) : (i += 1) {
                const c = counts[i];
                if (target < sum + c) {
                    const rows_before = if (by_lines) other else sum;
                    const lines_before = if (by_lines) sum else other;
                    const rb = rows_before - self.base_rows;
                    const lb = lines_before - self.base_lines;
                    return .{ .index = i - self.head, .rows_before = rb, .lines_before = lb };
                }
                sum += c;
                other += if (by_lines) self.rows_.items[i] else self.lines_.items[i];
            }
            return null;
        }

        /// Like `findLine`, but scans forward from `from` when `k` is near,
        /// which makes a sorted sweep linear. `k` below `from` falls back.
        pub fn seekLine(self: *const Self, from: Pos, k: u64) ?Pos {
            if (k < from.lines_before) return self.findLine(k);
            var pos = from;
            var steps: usize = 0;
            while (steps < block_entries) : (steps += 1) {
                const e = self.entry(pos.index);
                if (k < pos.lines_before + e.lines) return pos;
                pos = self.next(pos) orelse return null;
            }
            return self.findLine(k);
        }

        /// Position of the first live entry whose `keyOf` is at least `key`;
        /// `len()` when none. The keys must increase along the index. Costs
        /// one node read per halving, so it is fast even cold.
        pub fn lowerBound(self: *const Self, comptime keyOf: fn (Ptr) u64, key: u64) usize {
            var lo: usize = 0;
            var hi: usize = self.len();
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (keyOf(self.nodes.items[self.head + mid]) < key) lo = mid + 1 else hi = mid;
            }
            return lo;
        }

        /// The position of live entry `i`, from the block sums plus a scan of
        /// at most one block.
        pub fn posAt(self: *const Self, i: usize) ?Pos {
            if (i >= self.len()) return null;
            const a = self.head + i;
            const b = a / block_entries;
            var rb = self.pre_rows.items[b];
            var lb = self.pre_lines.items[b];
            for (b * block_entries..a) |j| {
                rb += self.rows_.items[j];
                lb += self.lines_.items[j];
            }
            return .{
                .index = i,
                .rows_before = rb - self.base_rows,
                .lines_before = lb - self.base_lines,
            };
        }

        /// The position after `pos`, or null at the end.
        pub fn next(self: *const Self, pos: Pos) ?Pos {
            if (pos.index + 1 >= self.len()) return null;
            const e = self.entry(pos.index);
            return .{
                .index = pos.index + 1,
                .rows_before = pos.rows_before + e.rows,
                .lines_before = pos.lines_before + e.lines,
            };
        }
    };
}

/// What a line edit changed.
pub const EditKind = enum(u8) {
    /// The line count of a region changed.
    recount,
    /// Lines kept their ids but their text was rewritten.
    rewrite,
};

/// One entry of the edit log. `old_lines`/`new_lines` are equal for a rewrite.
pub const LineEdit = struct {
    seq: u64,
    kind: EditKind,
    first_line: u64,
    old_lines: u64,
    new_lines: u64,
};

/// Edits kept in the ring.
pub const edit_ring = 16;

/// The last `edit_ring` edits, numbered from 1.
///
/// A reader that falls more than `edit_ring` edits behind loses some; that is
/// reported as an overflow until it reads up to the newest sequence again.
pub const EditLog = struct {
    ring: [edit_ring]LineEdit = undefined,
    /// Sequence of the newest edit; 0 when there is none.
    seq: u64 = 0,
    /// Highest `since_seq` any reader has passed.
    read_seq: u64 = 0,
    /// Highest sequence overwritten before it was read.
    lost_through: u64 = 0,

    /// Result of `since`.
    pub const Since = struct { n: usize, overflowed: bool };

    /// Record an edit and return its sequence.
    pub fn push(self: *EditLog, kind: EditKind, first_line: u64, old_lines: u64, new_lines: u64) u64 {
        if (self.seq >= edit_ring) {
            // The slot being reused holds the oldest retained edit.
            const dying = self.seq - edit_ring + 1;
            if (dying > self.read_seq) self.lost_through = @max(self.lost_through, dying);
        }
        self.seq += 1;
        self.ring[@intCast((self.seq - 1) % edit_ring)] = .{
            .seq = self.seq,
            .kind = kind,
            .first_line = first_line,
            .old_lines = old_lines,
            .new_lines = new_lines,
        };
        return self.seq;
    }

    /// Copy edits newer than `since_seq`, oldest first, into `out`. When some
    /// were lost, `overflowed` is set and the retained ones are still given.
    pub fn since(self: *EditLog, since_seq: u64, out: []LineEdit) Since {
        self.read_seq = @max(self.read_seq, since_seq);
        const oldest = if (self.seq > edit_ring) self.seq - edit_ring + 1 else 1;
        const start = @max(since_seq + 1, oldest);
        var n: usize = 0;
        var s = start;
        while (s <= self.seq and n < out.len) : (s += 1) {
            out[n] = self.ring[@intCast((s - 1) % edit_ring)];
            n += 1;
        }
        return .{ .n = n, .overflowed = self.lost_through > since_seq };
    }

    /// Record that something changed which no edit can describe: readers
    /// see an overflow until they read up to the new sequence.
    pub fn markLost(self: *EditLog) void {
        self.seq += 1;
        self.lost_through = self.seq;
        self.ring[@intCast((self.seq - 1) % edit_ring)] = .{
            .seq = self.seq,
            .kind = .recount,
            .first_line = 0,
            .old_lines = 0,
            .new_lines = 0,
        };
    }

    /// Whether an edit was lost that no reader has caught up past.
    pub fn overflowed(self: *const EditLog) bool {
        return self.lost_through > self.read_seq;
    }
};

// ------------------------------------------------------------------ tests

const testing = std.testing;
const test_wrap_bit: u6 = 5;

fn fillRaw(raw: []u64, rnd: std.Random, wrapped_per_mille: u32) void {
    for (raw) |*r| {
        // Noise in every other bit, so only the wrap bit may matter.
        var v = rnd.int(u64) & ~(@as(u64, 1) << test_wrap_bit);
        if (rnd.uintLessThan(u32, 1000) < wrapped_per_mille) v |= @as(u64, 1) << test_wrap_bit;
        r.* = v;
    }
}

test "LineIndex LineEnds build rank select agree with a naive scan" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    var raw: [700]u64 = undefined;
    // 0 = no row wraps, 1000 = all rows wrap, the rest mixed.
    const mixes = [_]u32{ 0, 1000, 500, 100, 900 };
    for (mixes) |mix| {
        var rows: usize = 1;
        while (rows <= 700) : (rows += if (rows < 70) 1 else 37) {
            fillRaw(raw[0..rows], rnd, mix);
            const le = try LineEnds.build(testing.allocator, raw[0..rows], test_wrap_bit);
            defer le.deinit();

            var naive: u32 = 0;
            for (raw[0..rows], 0..) |r, i| {
                try testing.expectEqual(naive, le.rank(@intCast(i)));
                const end = (r >> test_wrap_bit) & 1 == 0;
                if (end) {
                    try testing.expectEqual(@as(?u32, @intCast(i)), le.select(naive));
                    naive += 1;
                }
            }
            try testing.expectEqual(naive, le.count);
            try testing.expectEqual(naive, countLineEnds(raw[0..rows], test_wrap_bit));
            try testing.expectEqual(naive, le.rank(@intCast(rows)));
            try testing.expectEqual(@as(u32, @intCast(rows)), le.rows);
            try testing.expectEqual(@as(?u32, null), le.select(naive));
        }
    }

    // The last row wrapped is not a line end; the previous one is.
    raw = @splat(0);
    raw[9] = @as(u64, 1) << test_wrap_bit;
    const le = try LineEnds.build(testing.allocator, raw[0..10], test_wrap_bit);
    defer le.deinit();
    try testing.expectEqual(@as(u32, 9), le.count);
    try testing.expectEqual(@as(?u32, 8), le.select(8));
    try testing.expectEqual(@as(?u32, null), le.select(9));
}

const TestIndex = NodeIndex(u32);

/// A model of `TestIndex`: plain entries, checked by brute force.
fn expectIndexMatches(index: *const TestIndex, model: []const TestIndex.Entry) !void {
    try testing.expectEqual(model.len, index.len());
    var rows: u64 = 0;
    var lines: u64 = 0;
    for (model, 0..) |e, i| {
        try testing.expectEqual(e.node, index.entry(i).node);
        try testing.expectEqual(e.rows, index.entry(i).rows);
        try testing.expectEqual(e.lines, index.entry(i).lines);
        rows += e.rows;
        lines += e.lines;
    }
    try testing.expectEqual(rows, index.totalRows());
    try testing.expectEqual(lines, index.totalLines());

    // Every line and row target, with the entry a naive prefix sum names.
    var i: usize = 0;
    var rb: u64 = 0;
    var lb: u64 = 0;
    var k: u64 = 0;
    while (k < lines) : (k += 1) {
        while (k >= lb + model[i].lines) {
            rb += model[i].rows;
            lb += model[i].lines;
            i += 1;
        }
        const pos = index.findLine(k).?;
        try testing.expectEqual(i, pos.index);
        try testing.expectEqual(rb, pos.rows_before);
        try testing.expectEqual(lb, pos.lines_before);
    }
    try testing.expectEqual(@as(?TestIndex.Pos, null), index.findLine(lines));

    i = 0;
    rb = 0;
    lb = 0;
    var r: u64 = 0;
    while (r < rows) : (r += 1) {
        while (r >= rb + model[i].rows) {
            rb += model[i].rows;
            lb += model[i].lines;
            i += 1;
        }
        const pos = index.findRow(r).?;
        try testing.expectEqual(i, pos.index);
        try testing.expectEqual(rb, pos.rows_before);
        try testing.expectEqual(lb, pos.lines_before);
    }
    try testing.expectEqual(@as(?TestIndex.Pos, null), index.findRow(rows));
}

test "LineIndex NodeIndex append truncate splice dropFront and find agree with a naive prefix sum" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xbeef);
    const rnd = prng.random();

    var index: TestIndex = .{};
    defer index.deinit(alloc);
    var model: std.ArrayList(TestIndex.Entry) = .empty;
    defer model.deinit(alloc);

    var next_id: u32 = 0;
    const Gen = struct {
        fn entry(r: std.Random, id: *u32) TestIndex.Entry {
            id.* += 1;
            const rows = r.intRangeAtMost(u16, 1, 9);
            return .{ .node = id.*, .rows = rows, .lines = r.intRangeAtMost(u16, 0, rows) };
        }
    };

    // Several blocks, so the prefix sums and the head are really exercised.
    for (0..700) |_| {
        const e = Gen.entry(rnd, &next_id);
        try index.append(alloc, e.node, e.rows, e.lines);
        try model.append(alloc, e);
    }
    try expectIndexMatches(&index, model.items);

    for (0..40) |step| {
        switch (rnd.uintLessThan(u8, 5)) {
            0 => { // append a few
                for (0..rnd.intRangeAtMost(usize, 1, 80)) |_| {
                    const e = Gen.entry(rnd, &next_id);
                    try index.append(alloc, e.node, e.rows, e.lines);
                    try model.append(alloc, e);
                }
            },
            1 => { // truncate
                const from = rnd.intRangeAtMost(usize, model.items.len / 2, model.items.len);
                index.truncate(from);
                model.shrinkRetainingCapacity(from);
            },
            2 => { // splice: shrink, same size or grow
                const pos = rnd.intRangeAtMost(usize, 0, model.items.len);
                const remove = rnd.intRangeAtMost(usize, 0, @min(model.items.len - pos, 30));
                var fresh: [40]TestIndex.Entry = undefined;
                const add = rnd.intRangeAtMost(usize, 0, 40);
                for (fresh[0..add]) |*f| f.* = Gen.entry(rnd, &next_id);
                try index.splice(alloc, pos, remove, fresh[0..add]);
                try model.replaceRange(alloc, pos, remove, fresh[0..add]);
            },
            3 => { // drop from the front
                const n = rnd.intRangeAtMost(usize, 0, @min(model.items.len, 90));
                index.dropFront(n);
                model.replaceRangeAssumeCapacity(0, n, &.{});
            },
            else => { // keep it from draining
                for (0..120) |_| {
                    const e = Gen.entry(rnd, &next_id);
                    try index.append(alloc, e.node, e.rows, e.lines);
                    try model.append(alloc, e);
                }
            },
        }
        errdefer std.debug.print("step {d}\n", .{step});
        try expectIndexMatches(&index, model.items);
    }

    // posAt names the same place as a search for the entry's first row.
    {
        var rb: u64 = 0;
        for (model.items, 0..) |e, i| {
            if (i % 37 == 0) try testing.expectEqual(index.posAt(i).?.rows_before, rb);
            rb += e.rows;
        }
        try testing.expectEqual(@as(?TestIndex.Pos, null), index.posAt(model.items.len));
    }

    // seekLine from any earlier position agrees with findLine.
    if (index.totalLines() > 10) {
        var pos = index.findLine(0).?;
        var k: u64 = 0;
        while (k < index.totalLines()) : (k += rnd.intRangeAtMost(u64, 1, 900)) {
            pos = index.seekLine(pos, k).?;
            try testing.expectEqual(index.findLine(k).?, pos);
        }
        // Backwards falls back to a fresh search.
        try testing.expectEqual(index.findLine(3).?, index.seekLine(pos, 3).?);
    }

    // next walks every entry.
    var walked: usize = 0;
    var pos: ?TestIndex.Pos = if (index.len() > 0) index.findRow(0).? else null;
    while (pos) |p| : (pos = index.next(p)) {
        try testing.expectEqual(walked, p.index);
        walked += 1;
    }
    try testing.expectEqual(index.len(), walked);

    // Dropping everything leaves a usable, empty index.
    index.dropFront(index.len());
    try testing.expectEqual(@as(usize, 0), index.len());
    try testing.expectEqual(@as(?TestIndex.Pos, null), index.findRow(0));
    try index.append(alloc, 7, 3, 2);
    try testing.expectEqual(@as(u64, 3), index.totalRows());
    try testing.expectEqual(@as(usize, 0), index.findLine(1).?.index);
}

test "LineIndex NodeIndex same-length splices and block-aligned compaction agree with a naive prefix sum" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xc0de);
    const rnd = prng.random();
    var index: TestIndex = .{};
    defer index.deinit(alloc);
    var model: std.ArrayList(TestIndex.Entry) = .empty;
    defer model.deinit(alloc);

    var id: u32 = 0;
    const gen = struct {
        fn entry(r: std.Random, next: *u32) TestIndex.Entry {
            next.* += 1;
            const rows = r.intRangeAtMost(u16, 1, 9);
            return .{ .node = next.*, .rows = rows, .lines = r.intRangeAtMost(u16, 0, rows) };
        }
    };
    for (0..3000) |_| {
        const e = gen.entry(rnd, &id);
        try index.append(alloc, e.node, e.rows, e.lines);
        try model.append(alloc, e);
    }
    for (0..60) |_| {
        // Same-length splices, sometimes spanning several blocks.
        const n = if (rnd.boolean()) rnd.intRangeAtMost(usize, 0, 4) else rnd.intRangeAtMost(usize, 250, 600);
        const pos = rnd.intRangeAtMost(usize, 0, model.items.len - n);
        const fresh = try alloc.alloc(TestIndex.Entry, n);
        defer alloc.free(fresh);
        for (fresh) |*f| f.* = gen.entry(rnd, &id);
        try index.splice(alloc, pos, n, fresh);
        try model.replaceRange(alloc, pos, n, fresh);
        try expectIndexMatches(&index, model.items);

        // Drops that cross the compaction threshold, then refill.
        if (rnd.uintLessThan(u8, 4) == 0) {
            const d = rnd.intRangeAtMost(usize, 0, model.items.len / 2 + 300);
            const drop = @min(d, model.items.len);
            index.dropFront(drop);
            model.replaceRangeAssumeCapacity(0, drop, &.{});
            for (0..700) |_| {
                const e = gen.entry(rnd, &id);
                try index.append(alloc, e.node, e.rows, e.lines);
                try model.append(alloc, e);
            }
            try expectIndexMatches(&index, model.items);
        }
    }
}

test "LineIndex NodeIndex chunked compaction survives edits while it runs" {
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xfeed);
    const rnd = prng.random();
    var index: TestIndex = .{};
    defer index.deinit(alloc);
    var model: std.ArrayList(TestIndex.Entry) = .empty;
    defer model.deinit(alloc);

    var id: u32 = 0;
    const gen = struct {
        fn entry(r: std.Random, next: *u32) TestIndex.Entry {
            next.* += 1;
            const rows = r.intRangeAtMost(u16, 1, 9);
            return .{ .node = next.*, .rows = rows, .lines = r.intRangeAtMost(u16, 0, rows) };
        }
    };
    for (0..2_500) |_| {
        const e = gen.entry(rnd, &id);
        try index.append(alloc, e.node, e.rows, e.lines);
        try model.append(alloc, e);
    }
    var saw_moving = false;
    for (0..150) |_| {
        switch (rnd.uintLessThan(u8, 5)) {
            0, 1 => { // drop a little: advances a running compaction
                const d = @min(rnd.intRangeAtMost(usize, 0, 120), model.items.len / 2);
                index.dropFront(d);
                model.replaceRangeAssumeCapacity(0, d, &.{});
            },
            2 => { // append
                for (0..rnd.intRangeAtMost(usize, 20, 150)) |_| {
                    const e = gen.entry(rnd, &id);
                    try index.append(alloc, e.node, e.rows, e.lines);
                    try model.append(alloc, e);
                }
            },
            3 => { // splice, same length or not, anywhere
                const pos = rnd.intRangeAtMost(usize, 0, model.items.len - 6);
                const remove = rnd.intRangeAtMost(usize, 0, 5);
                var fresh: [6]TestIndex.Entry = undefined;
                const add = if (rnd.boolean()) remove else rnd.intRangeAtMost(usize, 0, 6);
                for (fresh[0..add]) |*f| f.* = gen.entry(rnd, &id);
                try index.splice(alloc, pos, remove, fresh[0..add]);
                try model.replaceRange(alloc, pos, remove, fresh[0..add]);
            },
            else => { // truncate a little, then refill
                const from = model.items.len - rnd.intRangeAtMost(usize, 0, 30);
                index.truncate(from);
                model.shrinkRetainingCapacity(from);
                for (0..60) |_| {
                    const e = gen.entry(rnd, &id);
                    try index.append(alloc, e.node, e.rows, e.lines);
                    try model.append(alloc, e);
                }
            },
        }
        saw_moving = saw_moving or index.moving;
        try expectIndexMatches(&index, model.items);
    }
    try testing.expect(saw_moving);
}

test "LineIndex EditLog since reports overflow after 16 unread edits" {
    var log: EditLog = .{};
    var out: [32]LineEdit = undefined;

    try testing.expectEqual(@as(usize, 0), log.since(0, &out).n);
    try testing.expect(!log.overflowed());

    for (1..17) |i| try testing.expectEqual(@as(u64, i), log.push(.recount, i * 10, 2, 3));
    const full = log.since(0, &out);
    try testing.expectEqual(@as(usize, 16), full.n);
    try testing.expect(!full.overflowed);
    try testing.expectEqual(@as(u64, 1), out[0].seq);
    try testing.expectEqual(@as(u64, 16), out[15].seq);
    try testing.expectEqual(@as(u64, 20), out[1].first_line);
    try testing.expectEqual(EditKind.recount, out[1].kind);

    // The 17th unread edit pushes the first one out.
    _ = log.push(.rewrite, 99, 4, 4);
    try testing.expect(log.overflowed());
    const lost = log.since(0, &out);
    try testing.expect(lost.overflowed);
    try testing.expectEqual(@as(usize, 16), lost.n);
    try testing.expectEqual(@as(u64, 2), out[0].seq);
    try testing.expectEqual(@as(u64, 17), out[15].seq);
    try testing.expectEqual(EditKind.rewrite, out[15].kind);

    // Reading up to the newest sequence clears the flag.
    const caught = log.since(17, &out);
    try testing.expectEqual(@as(usize, 0), caught.n);
    try testing.expect(!caught.overflowed);
    try testing.expect(!log.overflowed());

    // A reader that keeps up never overflows, however many edits pass.
    for (0..50) |_| {
        _ = log.push(.recount, 1, 1, 2);
        try testing.expectEqual(@as(usize, 1), log.since(log.seq - 1, &out).n);
    }
    try testing.expect(!log.overflowed());

    // A small output buffer returns the oldest edits first and no overflow.
    for (0..3) |_| _ = log.push(.recount, 1, 1, 2);
    const some = log.since(log.seq - 3, out[0..2]);
    try testing.expectEqual(@as(usize, 2), some.n);
    try testing.expect(!some.overflowed);
}

test "LineIndex EditLog markLost overflows until read" {
    var log: EditLog = .{};
    var out: [4]LineEdit = undefined;
    _ = log.push(.rewrite, 5, 1, 1);
    log.markLost();
    try testing.expect(log.overflowed());
    try testing.expect(log.since(1, &out).overflowed);
    try testing.expect(!log.since(2, &out).overflowed);
    try testing.expect(!log.overflowed());
}

test "LineIndex NodeIndex lowerBound finds entries by an increasing key" {
    const alloc = testing.allocator;
    const Key = struct {
        fn id(n: u32) u64 {
            return n;
        }
    };
    var index: TestIndex = .{};
    defer index.deinit(alloc);
    for (0..1000) |i| try index.append(alloc, @intCast(i * 3 + 1), 1, 1);
    index.dropFront(300);
    index.truncate(500);
    // Keys 901, 904, ... live at positions 0..; between them the next entry.
    for (0..500) |i| {
        const key: u64 = (i + 300) * 3 + 1;
        try testing.expectEqual(i, index.lowerBound(Key.id, key));
        try testing.expectEqual(i + 1, index.lowerBound(Key.id, key + 1));
    }
    try testing.expectEqual(@as(usize, 0), index.lowerBound(Key.id, 0));
    try testing.expectEqual(@as(usize, 500), index.lowerBound(Key.id, 100_000));
}
