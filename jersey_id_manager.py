"""
Jersey-based ID Manager for stable player tracking.

This module maps jersey numbers to stable track IDs to solve ID flickering.
When the same player (same jersey number) is detected with different ByteTrack IDs,
this manager ensures they get the same stable ID.

Key Features:
- Maps jersey number → stable ID (e.g., #23 always gets ID 1023)
- Resolves ID conflicts (same jersey detected with multiple track IDs)
- Handles team-scoped IDs (Team A: 1000s, Team B: 2000s, Referee: 3000s)
- Caches jersey detections to avoid OCR on every frame
"""

class JerseyBasedIDManager:
    def __init__(self):
        # Jersey number → stable ID mapping
        self.jersey_to_stable_id = {
            'Team A': {},  # {jersey_num: stable_id}
            'Team B': {},
            'Referee': {},
            'Unknown': {}
        }

        # Track ID → jersey number cache
        self.track_to_jersey = {}  # {track_id: {'jersey_num': int, 'team': str, 'last_seen': frame_idx}}

        # Next available stable ID for each team
        self.next_stable_id = {
            'Team A': 1000,  # Team A IDs: 1000-1999
            'Team B': 2000,  # Team B IDs: 2000-2999
            'Referee': 3000,  # Referee IDs: 3000-3999
            'Unknown': 4000  # Unknown IDs: 4000-4999
        }

        # Cache timeout (frames)
        self.cache_timeout = 90  # 3 seconds @ 30fps

        print("✓ JerseyBasedIDManager initialized")
        print("  Team A IDs: 1000-1999")
        print("  Team B IDs: 2000-2999")
        print("  Referee IDs: 3000-3999")

    def get_stable_id(self, track_id, jersey_num, team, frame_idx):
        """
        Get stable ID for a player based on jersey number.

        Args:
            track_id: Current ByteTrack ID
            jersey_num: Detected jersey number (int or None)
            team: Team name ('Team A', 'Team B', 'Referee', 'Unknown')
            frame_idx: Current frame index

        Returns:
            stable_id: Consistent ID based on jersey number
        """
        # Clean expired cache entries
        self._clean_cache(frame_idx)

        # If no jersey number detected, check cache
        if jersey_num is None:
            cached = self.track_to_jersey.get(track_id)
            if cached and frame_idx - cached['last_seen'] < self.cache_timeout:
                # Use cached jersey number
                jersey_num = cached['jersey_num']
                team = cached['team']
            else:
                # No jersey info - use track_id as-is
                return track_id

        # Update cache
        self.track_to_jersey[track_id] = {
            'jersey_num': jersey_num,
            'team': team,
            'last_seen': frame_idx
        }

        # Check if this jersey already has a stable ID
        team_jerseys = self.jersey_to_stable_id[team]

        if jersey_num in team_jerseys:
            # Existing jersey - return stable ID
            return team_jerseys[jersey_num]
        else:
            # New jersey - assign new stable ID
            stable_id = self.next_stable_id[team]
            self.next_stable_id[team] += 1

            team_jerseys[jersey_num] = stable_id

            print(f"  🆔 New player: Jersey #{jersey_num} ({team}) → Stable ID {stable_id}")

            return stable_id

    def resolve_conflicts(self, tracks, frame_idx):
        """
        Resolve ID conflicts in current frame.

        If multiple tracks have same jersey number, keep the one with highest confidence.
        If multiple tracks have same stable ID (shouldn't happen), keep first one.

        Args:
            tracks: List of track dicts with 'track_id', 'jersey_num', 'team', 'confidence'
            frame_idx: Current frame index

        Returns:
            Resolved tracks with stable IDs
        """
        # Group by jersey number
        jersey_groups = {}  # {(team, jersey_num): [tracks]}

        for track in tracks:
            jersey_num = track.get('jersey_num')
            team = track.get('team', 'Unknown')

            if jersey_num is not None:
                key = (team, jersey_num)
                if key not in jersey_groups:
                    jersey_groups[key] = []
                jersey_groups[key].append(track)

        # Resolve conflicts (multiple tracks with same jersey)
        resolved_tracks = []
        used_stable_ids = set()

        for (team, jersey_num), group_tracks in jersey_groups.items():
            if len(group_tracks) > 1:
                # Conflict! Multiple tracks with same jersey
                # Keep the one with highest confidence
                best_track = max(group_tracks, key=lambda t: t.get('confidence', 0))

                print(f"  ⚠️ Conflict resolved: Jersey #{jersey_num} ({team}) detected {len(group_tracks)} times, "
                      f"kept track #{best_track['track_id']} (conf={best_track['confidence']:.2%})")

                # Assign stable ID
                stable_id = self.get_stable_id(
                    best_track['track_id'], jersey_num, team, frame_idx
                )
                best_track['stable_id'] = stable_id
                used_stable_ids.add(stable_id)
                resolved_tracks.append(best_track)
            else:
                # No conflict - single track
                track = group_tracks[0]
                stable_id = self.get_stable_id(
                    track['track_id'], jersey_num, team, frame_idx
                )
                track['stable_id'] = stable_id
                used_stable_ids.add(stable_id)
                resolved_tracks.append(track)

        # Add tracks without jersey numbers (use original track_id)
        for track in tracks:
            if track.get('jersey_num') is None:
                track['stable_id'] = track['track_id']
                resolved_tracks.append(track)

        return resolved_tracks

    def _clean_cache(self, frame_idx):
        """Remove expired cache entries."""
        expired_ids = [
            tid for tid, data in self.track_to_jersey.items()
            if frame_idx - data['last_seen'] > self.cache_timeout
        ]

        for tid in expired_ids:
            del self.track_to_jersey[tid]

    def get_stats(self):
        """Get statistics about jersey-based tracking."""
        stats = {
            'total_unique_players': 0,
            'team_counts': {}
        }

        for team, jerseys in self.jersey_to_stable_id.items():
            count = len(jerseys)
            stats['team_counts'][team] = count
            stats['total_unique_players'] += count

        return stats
