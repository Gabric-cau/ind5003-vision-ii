"""Text can support a candidate, but OCR disagreement alone is not an identity veto."""
from src.camera_view_guard import closeup_guard


def text_camera_decision(old_vote,original_vote,query,people,context):
    result={'vote':old_vote,'changed':False,'warning':None,'reason':None,'identity_verified':False}
    if original_vote!='kicker_closeup':return result
    status=query['status']
    if status=='number_conflict':
        return {**result,'warning':'OCR_number_disagrees_with_manifest; repeated_font_errors_possible; review_only'}
    if status not in ('number_match_candidate','number_and_name_candidate','name_match_candidate'):return result
    gate=closeup_guard(people,context)
    if old_vote=='person_closeup_identity_unresolved' and gate['reason']=='insufficient_field_context_not_proof_of_spectator':
        return {**result,'vote':'kicker_closeup','changed':True,
                'reason':'repeated_text_support_relaxes_grass_only_gate',
                'warning':'Candidate only; repeated text and same-kit appearance do not prove identity.'}
    return result
