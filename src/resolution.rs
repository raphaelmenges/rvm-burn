pub struct Resolution {
    pub name: &'static str,
    pub src_width: usize,
    pub src_height: usize,
    pub r1_width: usize,
    pub r1_height: usize,
    pub r2_width: usize,
    pub r2_height: usize,
    pub r3_width: usize,
    pub r3_height: usize,
    pub r4_width: usize,
    pub r4_height: usize,
}

pub const FAST: Resolution = Resolution {
    name: "fast",
    src_width: 120,
    src_height: 90,
    r1_width: 60,
    r1_height: 45,
    r2_width: 30,
    r2_height: 23,
    r3_width: 15,
    r3_height: 12,
    r4_width: 8,
    r4_height: 6,
};

pub const BALANCED: Resolution = Resolution {
    name: "balanced",
    src_width: 160,
    src_height: 120,
    r1_width: 80,
    r1_height: 60,
    r2_width: 40,
    r2_height: 30,
    r3_width: 20,
    r3_height: 15,
    r4_width: 10,
    r4_height: 8,
};

pub const ACCURATE: Resolution = Resolution {
    name: "accurate",
    src_width: 320,
    src_height: 240,
    r1_width: 160,
    r1_height: 120,
    r2_width: 80,
    r2_height: 60,
    r3_width: 40,
    r3_height: 30,
    r4_width: 20,
    r4_height: 15,
};
