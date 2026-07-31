# Automatic Foreground Prompt V2 Design

## Scope

Replace only the automatic Claude Vision prompt used when the user supplies no
keywords. The manual-target/occluder prompt, output schemas, parsing, keyword
normalization, SAM3 segmentation, and all downstream mask, completion,
reconstruction, matting, grouping, and background workflows remain unchanged.

The automatic prompt must be standalone. It must not append
`COMMON_STRICT_RULES`, because the approved prompt contains its own complete and
mode-specific policy. `COMMON_STRICT_RULES` remains available to
`OCCLUDER_PROMPT`.

## Problem

The current automatic prompt can return background objects because it:

- describes foreground selection qualitatively without defining an editable
  layer;
- can treat two-dimensional overlap as evidence of occlusion;
- repeats shared rules that are not equally appropriate for automatic and
  manual modes;
- does not ask Claude to reconsider all instances that the final SAM3 keyword
  may match;
- does not explicitly prefer returning fewer high-precision keywords.

## Behavioral Requirements

The replacement prompt must:

1. Treat the task as editable foreground-layer selection, not captioning or a
   complete scene inventory.
2. Count visible people and choose `person`, `man`, `woman`, `boy`, `girl`, or
   `people` in the same decision step. `people` is allowed only when more than
   three people are visible.
3. Keep a clear, important object eligible as a primary layer even when it is
   behind another object. Being behind is not equivalent to being background.
4. Recognize an occluder only when it is closer to the camera and physically
   hides the target. Bounding-box overlap, silhouette overlap, touching, and
   proximity are insufficient.
5. Include genuine small occluders and support layered occlusion such as
   `C -> A -> B`.
6. Return whole semantic objects with their structural supports.
7. Attach worn and held items to the parent subject keyword instead of
   returning them separately.
8. Prefer meaningful parent collections over enumerating many small contents.
9. Exclude deep-background scenery, architecture and places, reflections,
   printed/screen-only objects, effects, textures, and uncertain fragments.
10. Use concise SAM3-friendly vocabulary and never invent modifiers.
11. Reconsider every visible instance that the final keyword may match.
12. First attempt a reliable, foreground-specific refinement when the same
    label occurs at different depths.
13. Veto a keyword only when an unwanted matching instance is simultaneously:
    truly deep-background, truly large or comparable in scale, heavily
    occluded by several objects/layers, and impossible to isolate with a safe
    refined keyword.
14. Preserve an otherwise valid foreground keyword when only the strict
    background-instance veto is uncertain.
15. Return at most ten unique keywords, preserve genuine occluders before
    low-value secondary objects, and never fill the quota with uncertain
    objects.

## Data Flow

The request and result contracts do not change:

```text
no user keywords
    -> FOREGROUND_OBJECT_PROMPT V2
    -> existing FOREGROUND_OUTPUT_SCHEMA
    -> existing JSON parsing and normalize_keywords()
    -> existing SAM3 segmentation
    -> unchanged downstream pipeline
```

The output remains:

```json
{
  "visible_person_count": 0,
  "keywords": []
}
```

## Approved Prompt

```text
Analyze the image and return concise English object labels for SAM3
segmentation. Each returned object may become a separate editable foreground
layer.

This is NOT an image-captioning or scene-inventory task. Select only objects
that are useful as independent editable layers. Prefer precision over recall:
return fewer keywords rather than include uncertain or background objects.

Follow these rules:

1. Count and label visible people

Count all real visible people or characters participating in the depicted
scene.

Do not count people appearing only inside posters, photographs, paintings,
screens, mirrors, or reflections within the scene.

Apply these naming rules while selecting person keywords:
- With one to three visible people, never use "people".
- Use "man", "woman", "boy", or "girl" only when clearly supported visually.
- Otherwise use "person".
- Use "people" only when more than three people are visible.
- Do not combine one to three distinct people into the general label "people".
- A visible person must still pass all foreground, occlusion, and global
  instance checks before being returned as a keyword.

2. Select primary editable objects

A primary object should:
- be clearly identifiable from its visible pixels;
- be visually important to the composition;
- be sufficiently visible to form a useful independent layer;
- belong to the foreground or a meaningful subject plane rather than the
  distant or deep background;
- have enough reliable visual evidence for a specific SAM3 label.

Object size alone is not sufficient. Large environmental regions such as the
sky, ground, road, water, mountains, walls, or distant vegetation remain
background.

An object located behind another object may still qualify as a primary object
when it remains clear, important, and sufficiently visible. Being behind
another object does not automatically make it background.

Exclude an occluded object only when it is reduced to a small ambiguous
fragment, its identity is uncertain, or it belongs to a deeply buried
background layer. A clear and important subject may still be selected when
partially occluded.

3. Add genuine occluders

For every selected primary object, include independent visible objects that
genuinely occlude it.

An object is an occluder only when:
- it is closer to the camera than the target; and
- it physically hides part of the target in the current image.

Bounding-box overlap, silhouette overlap, touching boundaries, or proximity
alone are NOT sufficient evidence of occlusion.

An object behind the target is never an occluder of that target, even when
their image regions overlap.

Examples:
- A distant tree behind a person is not an occluder of the person.
- A person standing in front of and hiding part of a car is an occluder of
  the car.
- A chair behind a person may still be a primary object, but it is not an
  occluder of the person.

Include a genuine occluder even when it is small or would otherwise be an
incidental object.

For layered occlusion such as C in front of A and A in front of B:
- include A when A hides part of B;
- include C when C hides A and also participates in the visible occlusion
  stack over B;
- never infer this relation from image overlap alone.

4. Return complete semantic objects

Return the complete parent object rather than isolated structural parts.

Include essential supports, stands, tripods, mounts, poles, bases, wheels,
mirrors, and directly attached structural components with their parent object.

Examples:
- Return "camera" for a camera together with its tripod or direct mount.
- Return "lamp" for a lamp together with its pole and base.
- Return "monitor" for a monitor together with its stand.
- Return a complete vehicle including its wheels, mirrors, and attached mounts.

Never return isolated hands, arms, legs, faces, wheels, poles, bases, or other
structural sub-components as separate keywords.

5. Handle people, characters, animals, and accessories

Return the complete person, character, or animal.

Treat clothing, footwear, glasses, jewelry, headwear, and worn accessories as
part of their parent subject.

When a person is wearing a hat, use a concise phrase such as:
- "man with hat";
- "woman with hat";
- "boy with hat";
- "girl with hat";
- "person with hat".

When a subject is holding, carrying, or wielding an item, include the item in
the subject label rather than returning it separately.

Use labels such as:
- "person holding umbrella";
- "man with sword";
- "woman with bag";
- "monkey with staff".

Do not also return the umbrella, sword, bag, staff, tool, phone, cup, weapon,
or other held item as a separate keyword.

A standalone item that is not worn or held may be returned independently only
when it qualifies as a clear and important primary foreground object.

6. Handle collections and contained objects

For a bag, basket, cart, suitcase, box, shelf, tray, rack, pile, or display,
prefer the meaningful parent collection instead of enumerating many small
contents.

Return an individual contained object only when it is visually important,
clearly independent, and useful as a separate editable layer.

Return a whole plant, tree, pot, dish, or meal rather than separate leaves,
branches, fruit, ingredients, toppings, or pieces.

7. Apply strict exclusions

Never return:
- distant or deep-background objects;
- scenery or environmental regions;
- tiny incidental objects unless they genuinely occlude a selected object;
- decorations, textures, patterns, shadows, highlights, or lighting effects;
- reflections or objects visible only through a reflection;
- objects appearing only inside posters, photographs, paintings, or screens;
- uncertain object-like regions;
- isolated body parts, clothing, wearables, or structural components.

Never return buildings, architectural structures, landmarks, venues,
locations, or places, even when they are large, visually dominant, close to
the camera, or appear to overlap another object.

This prohibition includes temples, pagodas, churches, cathedrals, shrines,
monuments, towers, castles, houses, huts, pavilions, palaces, skyscrapers,
bridges, gates, walls, rooms, venues, parks, and similar architectural places.

8. Use concise SAM3-friendly labels

Use clear, natural, standard English object names.

Normally use 1-3 words. A slightly longer phrase is allowed only when required
to keep a worn or held item attached to its parent subject.

Avoid colors, materials, decorative adjectives, and unnecessary modifiers by
default.

A concise modifier may be used only when:
- it is clearly visible and reliable;
- it preserves the exact object type;
- it is necessary to distinguish the intended foreground instance from other
  matching instances;
- it is likely to help SAM3 isolate the intended object.

Never invent or guess a modifier merely to preserve a candidate keyword.

Use the most specific label clearly supported by the image. Do not generalize
a specific object into a broader category.

9. Reconsider all matching instances

Before returning a keyword, inspect the entire image and reconsider every
visible instance that SAM3 may match, not only the clearest foreground one.

If the same label appears at different depth layers, first try a concise,
visually reliable refinement that focuses on the intended foreground instance.
Preserve the object type and never invent attributes. Do not use unreliable
spatial phrases such as "foreground car", "front person", or "nearest chair".
After refinement, check all matching instances again.

Omit the keyword only when an unwanted matching instance satisfies ALL of
these conditions:
- it is truly in the deep background;
- it is truly large or comparable in scale to the intended foreground
  instance;
- it is heavily occluded by several independent objects or occlusion layers;
- no reliable refined keyword can isolate the intended foreground instance.

Do not omit a keyword because of small distant instances, a large but clearly
visible background instance, or an instance covered by only one ordinary
occluder.

If the candidate object itself is uncertain, omit it according to the earlier
selection rules. If only the strict background-instance veto conditions are
uncertain, do not veto an otherwise valid foreground keyword.

10. Rank and limit the output

Return at most 10 unique keywords.

Order the output as follows:
1. Primary objects, ordered by visual importance.
2. Required occluders, ordered by how strongly they cover a retained primary
   object.

If one object is both a primary object and an occluder, return it only once at
its primary-object position.

When the keyword limit is reached:
- remove uncertain and low-importance secondary objects first;
- preserve genuine occluders of retained primary objects;
- never fill unused positions with background or uncertain objects.

Do not attempt to fill the quota. Returning fewer accurate keywords is better
than returning additional uncertain keywords.

Return JSON only and follow the provided output schema exactly.
```

## Error Handling and Compatibility

- Structured-output refusal, truncation, malformed JSON, invalid person count,
  invalid keyword types, keyword length, and keyword quota handling remain
  unchanged.
- The feature does not add a second VLM call or modify the response schema.
- Manual target refinement and per-target occluder extraction are explicitly
  out of scope.
- Existing unrelated manual-target validation regressions are not part of this
  prompt-only change.

## Verification

Focused tests should verify that automatic mode:

- sends the standalone V2 prompt without appending the shared common block;
- defines a genuine occluder by depth and physical hiding, not overlap;
- includes layered occlusion;
- combines visible-person counting with person-label selection;
- requires global instance reconsideration and the four-way conservative veto;
- preserves the existing automatic output schema and parsing behavior.

Manual-mode prompt and schema assertions should continue to pass unchanged.
