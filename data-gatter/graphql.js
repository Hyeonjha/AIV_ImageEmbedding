import { gql } from 'graphql-request';

export const queryLabelSetIdsInVersion = gql`
query labelSetIdsInVersion {
  labelSetIdsInVersion(where: { versionId: { equals: 18165 } })
}
`;
export const queryLabelSet = gql`
  query LabelItems($id: Int) {
    labelSet(filter: { id: $id }) {
        imageId
        Image {
            ImageServer {
                hash
                name
                type
                url
            }
            rawPath
            path
            width
            height
        }
        LabelItem {
            id
            points
            LabelClass {
                name
            }
        }
    }
  }
`;